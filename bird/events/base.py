import time
from typing import Any


class Event:
    name = "event"
    cooldown = 0

    def __init__(self):
        self.last_triggered = {}

    def detect(self, state: dict) -> list[dict]:
        """
        Return list of events:
        {
            "type": str,
            "confidence": float (0-1),
            "objects": list[int],
            "meta": dict
        }
        """
        raise NotImplementedError

    def _is_on_cooldown(self, event_id: str) -> bool:
        if self.cooldown <= 0:
            return False

        if event_id not in self.last_triggered:
            return False

        return (time.time() - self.last_triggered[event_id]) < self.cooldown

    def _mark_triggered(self, event_id: str):
        self.last_triggered[event_id] = time.time()

    def _point_in_polygon(self, point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def _bbox_center(self, bbox: list[float]) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
