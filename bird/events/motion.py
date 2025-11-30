from bird.events.base import Event


class RegionEntryEvent(Event):
    name = "entry"

    def __init__(self, region: list[tuple[float, float]], cooldown: float = 2.0):
        super().__init__()
        self.region = region
        self.cooldown = cooldown
        self.inside_region = set()

    def detect(self, state: dict) -> list[dict]:
        events = []
        tracked_objects = state.get("tracked_objects", [])

        current_inside = set()

        for obj in tracked_objects:
            track_id = obj.get("track_id")
            bbox = obj.get("bbox")

            if not bbox or track_id is None:
                continue

            center = self._bbox_center(bbox)
            is_inside = self._point_in_polygon(center, self.region)

            if is_inside:
                current_inside.add(track_id)

                if track_id not in self.inside_region:
                    event_id = f"entry_{track_id}"
                    if not self._is_on_cooldown(event_id):
                        events.append({
                            "type": "region_entry",
                            "confidence": obj.get("confidence", 0.9),
                            "objects": [track_id],
                            "meta": {
                                "class": obj.get("class", "unknown"),
                                "position": center
                            }
                        })
                        self._mark_triggered(event_id)

        self.inside_region = current_inside
        return events


class RegionExitEvent(Event):
    name = "exit"

    def __init__(self, region: list[tuple[float, float]], cooldown: float = 2.0):
        super().__init__()
        self.region = region
        self.cooldown = cooldown
        self.inside_region = set()

    def detect(self, state: dict) -> list[dict]:
        events = []
        tracked_objects = state.get("tracked_objects", [])

        current_inside = set()

        for obj in tracked_objects:
            track_id = obj.get("track_id")
            bbox = obj.get("bbox")

            if not bbox or track_id is None:
                continue

            center = self._bbox_center(bbox)
            is_inside = self._point_in_polygon(center, self.region)

            if is_inside:
                current_inside.add(track_id)

        for track_id in self.inside_region:
            if track_id not in current_inside:
                event_id = f"exit_{track_id}"
                if not self._is_on_cooldown(event_id):
                    events.append({
                        "type": "region_exit",
                        "confidence": 0.9,
                        "objects": [track_id],
                        "meta": {}
                    })
                    self._mark_triggered(event_id)

        self.inside_region = current_inside
        return events