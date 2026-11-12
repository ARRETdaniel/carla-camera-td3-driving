"""M1 (2026-10-16): the CARLA environment in its own process, with every call bounded by a time-out."""

import importlib
import logging
import multiprocessing as mp
import os
import random
import sys
import time
import traceback
from types import SimpleNamespace
from typing import Any, Dict, Optional

import numpy as np

TIMEOUT_REASON = "env_process_timeout"
_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

logger = logging.getLogger(__name__)


def _worker(conn, factory: str, kwargs: Dict[str, Any], seed: int, log_level: int, sys_path):
    for p in reversed(sys_path):
        if p not in sys.path:
            sys.path.insert(0, p)
    logging.basicConfig(level=log_level, format=_LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S", force=True)
    logging.getLogger().setLevel(log_level)
    np.random.seed(seed % (2 ** 32))
    random.seed(seed)
    try:
        module_name, attr = factory.split(":")
        env = getattr(importlib.import_module(module_name), attr)(**kwargs)
        conn.send(("ok", (env.observation_space, env.action_space)))
    except BaseException:
        conn.send(("error", traceback.format_exc()))
        conn.close()
        return
    while True:
        try:
            cmd, arg = conn.recv()
        except (EOFError, OSError):
            break
        try:
            if cmd == "step":
                result = env.step(arg)
            elif cmd == "reset":
                result = env.reset(**arg)
            elif cmd == "vehicle":
                v = getattr(env, "vehicle", None)
                result = None if v is None else (int(v.id), bool(v.is_alive))
            elif cmd == "close":
                env.close()
                conn.send(("ok", None))
                break
            else:
                raise ValueError(f"unknown command {cmd!r}")
            conn.send(("ok", result))
        except BaseException:
            conn.send(("error", traceback.format_exc()))
    conn.close()


class IsolatedEnv:

    def __init__(
        self,
        factory: str,
        kwargs: Dict[str, Any],
        seed: int = 0,
        step_timeout: float = 60.0,
        reset_timeout: float = 300.0,
        start_timeout: float = 300.0,
        max_restarts: int = 5,
    ):
        self.factory, self.kwargs, self.seed = factory, dict(kwargs), int(seed)
        self.step_timeout, self.reset_timeout = float(step_timeout), float(reset_timeout)
        self.start_timeout, self.max_restarts = float(start_timeout), int(max_restarts)
        self.restarts = 0
        self.observation_space = self.action_space = None
        self._ctx = mp.get_context("spawn")
        self._proc = self._conn = None
        self._last_obs = None
        self._closed = False
        self._start()

    def _start(self) -> None:
        parent_conn, child_conn = self._ctx.Pipe()
        self._proc = self._ctx.Process(
            target=_worker,
            args=(child_conn, self.factory, self.kwargs, self.seed + 1000 * self.restarts,
                  logging.getLogger().getEffectiveLevel(), list(sys.path)),
            daemon=True, name=f"isolated-env-{self.restarts}")
        self._proc.start()
        child_conn.close()
        self._conn = parent_conn
        res = self._wait(self.start_timeout)
        if res is None or res[0] != "ok":
            detail = "time-out" if res is None else res[1]
            self._kill()
            raise RuntimeError(f"isolated environment failed to start ({self.factory}): {detail}")
        self.observation_space, self.action_space = res[1]
        logger.warning(f"[M1] environment process started: pid {self._proc.pid}, restarts so far {self.restarts}, "
                       f"step/reset time-outs {self.step_timeout:.0f}/{self.reset_timeout:.0f} s")

    def _wait(self, timeout: float):
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            if self._conn.poll(min(remaining, 1.0)):
                try:
                    return self._conn.recv()
                except (EOFError, OSError):
                    return ("died", f"pipe closed (exit code {self._proc.exitcode})")
            if not self._proc.is_alive():
                if self._conn.poll(0):
                    continue
                return ("died", f"process exited with code {self._proc.exitcode}")

    def _send(self, msg) -> bool:
        try:
            self._conn.send(msg)
            return True
        except (BrokenPipeError, EOFError, OSError):
            return False

    def _kill(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
        if self._proc is not None and self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(10)
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join(10)

    def _restart(self, what: str) -> None:
        self.restarts += 1
        pid = self._proc.pid
        wchan = ""
        try:
            with open(f"/proc/{pid}/wchan") as f:
                wchan = f.read().strip()
        except Exception:
            pass
        logger.error(f"[M1] environment process {pid} did not answer {what}; wait channel {wchan or 'n/a'}; "
                     f"killing it and starting restart {self.restarts}/{self.max_restarts}")
        self._kill()
        if self.restarts > self.max_restarts:
            raise RuntimeError(f"[M1] more than {self.max_restarts} environment restarts in one run: not transient")
        self._start()

    def step(self, action):
        res = self._wait(self.step_timeout) if self._send(("step", np.asarray(action))) else ("died", "pipe broken")
        if res is None or res[0] == "died":
            self._restart("step" if res is None else f"step ({res[1]})")
            info = {"termination_reason": TIMEOUT_REASON, "env_process_restarts": self.restarts}
            return self._last_obs, 0.0, False, True, info
        if res[0] == "error":
            raise RuntimeError(f"environment step raised in the child process:\n{res[1]}")
        self._last_obs = res[1][0]
        return res[1]

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        kwargs = {}
        if seed is not None:
            kwargs["seed"] = seed
        if options is not None:
            kwargs["options"] = options
        for attempt in (1, 2):
            res = self._wait(self.reset_timeout) if self._send(("reset", kwargs)) else ("died", "pipe broken")
            if res is None or res[0] == "died":
                self._restart(f"reset, attempt {attempt}" + ("" if res is None else f" ({res[1]})"))
                continue
            if res[0] == "error":
                raise RuntimeError(f"environment reset raised in the child process:\n{res[1]}")
            self._last_obs = res[1][0]
            return res[1]
        raise RuntimeError("[M1] environment reset did not answer twice in a row")

    @property
    def vehicle(self):
        if not self._send(("vehicle", None)):
            return None
        res = self._wait(30.0)
        if res is None or res[0] != "ok" or res[1] is None:
            return None
        return SimpleNamespace(id=res[1][0], is_alive=res[1][1])

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._proc is not None and self._proc.is_alive() and self._send(("close", None)):
                self._wait(60.0)
        except Exception:
            pass
        finally:
            self._kill()

    @property
    def is_closed(self) -> bool:
        return self._closed
