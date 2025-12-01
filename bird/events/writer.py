from pathlib import Path
from datetime import datetime


class EventWriter:
    def __init__(self, session_dir: str = None, enable_database: bool = True, enable_embeddings: bool = True):
        if session_dir is None:
            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            session_dir = f"sessions/{timestamp}"

        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.database = None
        if enable_database:
            from bird.events.database import EventDatabase
            self.database = EventDatabase(session_dir=str(self.session_dir))

        self.embedder = None
        if enable_embeddings:
            from bird.events.embedder import EventEmbedder
            self.embedder = EventEmbedder()

    def write_event(self, frame: int, timestamp: float, events: list[dict]):
        if not events:
            return

        if self.database:
            event_ids = self.database.write_events(frame, timestamp, events)

            if self.embedder and event_ids:
                embeddings = self.embedder.embed_events(events)
                for event_id, embedding in zip(event_ids, embeddings):
                    self.database.write_embedding(event_id, embedding, model=self.embedder.model_name)

    def write_scene_graph(self, frame: int, timestamp: float, scene_graph: dict):
        pass

    def get_session_path(self) -> Path:
        return self.session_dir
