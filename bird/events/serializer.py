import json
import os
from pathlib import Path
from datetime import datetime


class EventSerializer:
    def __init__(self, session_dir: str = None):
        if session_dir is None:
            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            session_dir = f"sessions/{timestamp}"

        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.events_file = self.session_dir / "events.jsonl"
        self.scene_graph_file = self.session_dir / "scene_graph.jsonl"

    def write_event(self, frame: int, timestamp: float, events: list[dict]):
        if not events:
            return

        event_entry = {
            "frame": frame,
            "timestamp": timestamp,
            "events": events
        }

        with open(self.events_file, 'a') as f:
            f.write(json.dumps(event_entry) + '\n')

    def write_scene_graph(self, frame: int, timestamp: float, scene_graph: dict):
        if not scene_graph:
            return

        scene_graph_entry = {
            "frame": frame,
            "timestamp": timestamp,
            "scene_graph": scene_graph
        }

        with open(self.scene_graph_file, 'a') as f:
            f.write(json.dumps(scene_graph_entry) + '\n')

    def get_session_path(self) -> Path:
        return self.session_dir

    def read_events(self):
        if not self.events_file.exists():
            return []

        events = []
        with open(self.events_file, 'r') as f:
            for line in f:
                events.append(json.loads(line))
        return events

    def read_scene_graphs(self):
        if not self.scene_graph_file.exists():
            return []

        scene_graphs = []
        with open(self.scene_graph_file, 'r') as f:
            for line in f:
                scene_graphs.append(json.loads(line))
        return scene_graphs
