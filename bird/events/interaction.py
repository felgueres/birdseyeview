import numpy as np
from bird.events.base import Event


class PersonObjectInteractionEvent(Event):
    name = "person_object_interaction"

    def __init__(self, distance_threshold: float = 50.0, duration_threshold: float = 2.0, cooldown: float = 3.0):
        super().__init__()
        self.distance_threshold = distance_threshold
        self.duration_threshold = duration_threshold
        self.cooldown = cooldown
        self.interaction_start_times = {}

    def detect(self, state: dict) -> list[dict]:
        events = []
        tracked_objects = state.get("tracked_objects", [])
        current_time = state.get("timestamp", 0)

        people = [obj for obj in tracked_objects if obj.get("class") == "person"]
        objects = [obj for obj in tracked_objects if obj.get("class") != "person"]

        current_interactions = set()

        for person in people:
            person_id = person.get("track_id")
            person_bbox = person.get("bbox")

            if not person_bbox or person_id is None:
                continue

            person_center = np.array(self._bbox_center(person_bbox))

            for obj in objects:
                obj_id = obj.get("track_id")
                obj_bbox = obj.get("bbox")

                if not obj_bbox or obj_id is None:
                    continue

                obj_center = np.array(self._bbox_center(obj_bbox))
                distance = np.linalg.norm(person_center - obj_center)

                interaction_key = (person_id, obj_id)

                if distance < self.distance_threshold:
                    current_interactions.add(interaction_key)

                    if interaction_key not in self.interaction_start_times:
                        self.interaction_start_times[interaction_key] = current_time

                    duration = current_time - self.interaction_start_times[interaction_key]

                    if duration >= self.duration_threshold:
                        event_id = f"interaction_{person_id}_{obj_id}"
                        if not self._is_on_cooldown(event_id):
                            events.append({
                                "type": "person_object_interaction",
                                "confidence": 0.85,
                                "objects": [person_id, obj_id],
                                "meta": {
                                    "distance": float(distance),
                                    "duration": float(duration),
                                    "object_class": obj.get("class", "unknown")
                                }
                            })
                            self._mark_triggered(event_id)

        for interaction_key in list(self.interaction_start_times.keys()):
            if interaction_key not in current_interactions:
                del self.interaction_start_times[interaction_key]

        return events
