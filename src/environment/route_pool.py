"""Route pool for multi-route training (v2, I1; 2026-10-24). Default OFF: the env keeps its single route."""
import os
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np


class RoutePool:

    def __init__(
        self,
        files: Sequence[str],
        build_manager: Callable[[str], object],
        build_profile: Optional[Callable[[Sequence], object]] = None,
        seed: Optional[int] = None,
    ):
        files = [str(f) for f in files]
        if not files:
            raise ValueError("route.pool.files is empty")
        if len(set(files)) != len(files):
            raise ValueError(f"route.pool.files has duplicates: {files}")
        self.entries: List[Dict] = []
        for i, f in enumerate(files):
            if not os.path.isfile(f):
                raise FileNotFoundError(f"route.pool file not found: {f}")
            manager = build_manager(f)
            profile = build_profile(manager.waypoints) if build_profile is not None else None
            self.entries.append({"index": i, "name": os.path.splitext(os.path.basename(f))[0], "file": f,
                                 "manager": manager, "profile": profile})
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.entries)

    def draw(self) -> Dict:
        return self.entries[int(self._rng.integers(len(self.entries)))]
