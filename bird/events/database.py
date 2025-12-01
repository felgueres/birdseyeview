import sqlite3
import json
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import contextmanager


class EventDatabase:
    def __init__(self, db_path: str = None, session_dir: str = None):
        if db_path is None:
            if session_dir is None:
                raise ValueError("Either db_path or session_dir must be provided")
            session_path = Path(session_dir)
            session_path.mkdir(parents=True, exist_ok=True)
            db_path = str(session_path / "events.db")

        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    frame INTEGER NOT NULL,
                    timestamp REAL NOT NULL,
                    type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    objects TEXT NOT NULL,
                    meta TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_frame ON events(frame)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON events(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_type ON events(type)")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    model TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (event_id) REFERENCES events(id)
                )
            """)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_event_id ON embeddings(event_id)")

            conn.commit()

    def write_event(self, frame: int, timestamp: float, event: Dict[str, Any]) -> int:
        with self._get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO events (frame, timestamp, type, confidence, objects, meta)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                frame,
                timestamp,
                event['type'],
                event['confidence'],
                json.dumps(event.get('objects', [])),
                json.dumps(event.get('meta', {}))
            ))
            conn.commit()
            return cursor.lastrowid

    def write_events(self, frame: int, timestamp: float, events: List[Dict[str, Any]]) -> List[int]:
        event_ids = []
        for event in events:
            event_id = self.write_event(frame, timestamp, event)
            event_ids.append(event_id)
        return event_ids

    def write_embedding(self, event_id: int, embedding: np.ndarray, model: str = "default"):
        embedding_bytes = embedding.astype(np.float32).tobytes()
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO embeddings (event_id, embedding, model)
                VALUES (?, ?, ?)
            """, (event_id, embedding_bytes, model))
            conn.commit()

    def get_events(
        self,
        event_type: Optional[str] = None,
        min_confidence: Optional[float] = None,
        frame_start: Optional[int] = None,
        frame_end: Optional[int] = None,
        timestamp_start: Optional[float] = None,
        timestamp_end: Optional[float] = None,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM events WHERE 1=1"
        params = []

        if event_type is not None:
            query += " AND type = ?"
            params.append(event_type)

        if min_confidence is not None:
            query += " AND confidence >= ?"
            params.append(min_confidence)

        if frame_start is not None:
            query += " AND frame >= ?"
            params.append(frame_start)

        if frame_end is not None:
            query += " AND frame <= ?"
            params.append(frame_end)

        if timestamp_start is not None:
            query += " AND timestamp >= ?"
            params.append(timestamp_start)

        if timestamp_end is not None:
            query += " AND timestamp <= ?"
            params.append(timestamp_end)

        query += " ORDER BY timestamp ASC, frame ASC"

        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        return self._rows_to_events(rows)

    def search_events_by_metadata(
        self,
        meta_key: str,
        meta_value: Any,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM events ORDER BY timestamp ASC, frame ASC"
            )
            rows = cursor.fetchall()

        events = []
        for row in rows:
            meta = json.loads(row['meta'])
            if meta_key in meta and meta[meta_key] == meta_value:
                events.append(self._row_to_event(row))
                if limit and len(events) >= limit:
                    break

        return events

    def get_events_by_object(
        self,
        object_id: int,
        event_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM events ORDER BY timestamp ASC, frame ASC"
            )
            rows = cursor.fetchall()

        events = []
        for row in rows:
            if event_type and row['type'] != event_type:
                continue

            objects = json.loads(row['objects'])
            if object_id in objects:
                events.append(self._row_to_event(row))

        return events

    def get_embedding(self, event_id: int, model: str = "default") -> Optional[np.ndarray]:
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT embedding FROM embeddings
                WHERE event_id = ? AND model = ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (event_id, model))
            row = cursor.fetchone()

        if row is None:
            return None

        return np.frombuffer(row['embedding'], dtype=np.float32)

    def get_all_embeddings(self, model: str = "default") -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT event_id, embedding FROM embeddings
                WHERE model = ?
                ORDER BY event_id ASC
            """, (model,))
            rows = cursor.fetchall()

        results = []
        for row in rows:
            results.append({
                'event_id': row['event_id'],
                'embedding': np.frombuffer(row['embedding'], dtype=np.float32)
            })

        return results

    def search_by_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        model: str = "default"
    ) -> List[Dict[str, Any]]:
        embeddings_data = self.get_all_embeddings(model=model)
        if not embeddings_data:
            return []

        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

        similarities = []
        for item in embeddings_data:
            emb = item['embedding']
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            similarity = np.dot(query_norm, emb_norm)
            similarities.append({
                'event_id': item['event_id'],
                'similarity': float(similarity)
            })

        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        top_results = similarities[:top_k]

        event_ids = [r['event_id'] for r in top_results]
        with self._get_connection() as conn:
            placeholders = ','.join('?' * len(event_ids))
            cursor = conn.execute(
                f"SELECT * FROM events WHERE id IN ({placeholders})",
                event_ids
            )
            rows = cursor.fetchall()

        id_to_event = {row['id']: self._row_to_event(row) for row in rows}

        results = []
        for result in top_results:
            event_id = result['event_id']
            if event_id in id_to_event:
                event = id_to_event[event_id]
                event['similarity'] = result['similarity']
                results.append(event)

        return results

    def get_event_statistics(self) -> Dict[str, Any]:
        with self._get_connection() as conn:
            total_cursor = conn.execute("SELECT COUNT(*) as count FROM events")
            total = total_cursor.fetchone()['count']

            type_cursor = conn.execute("""
                SELECT type, COUNT(*) as count
                FROM events
                GROUP BY type
                ORDER BY count DESC
            """)
            by_type = {row['type']: row['count'] for row in type_cursor.fetchall()}

            time_cursor = conn.execute("""
                SELECT MIN(timestamp) as min_time, MAX(timestamp) as max_time
                FROM events
            """)
            time_row = time_cursor.fetchone()

        return {
            'total_events': total,
            'by_type': by_type,
            'time_range': {
                'start': time_row['min_time'],
                'end': time_row['max_time']
            }
        }

    def _row_to_event(self, row) -> Dict[str, Any]:
        return {
            'id': row['id'],
            'frame': row['frame'],
            'timestamp': row['timestamp'],
            'type': row['type'],
            'confidence': row['confidence'],
            'objects': json.loads(row['objects']),
            'meta': json.loads(row['meta']),
            'created_at': row['created_at']
        }

    def _rows_to_events(self, rows) -> List[Dict[str, Any]]:
        return [self._row_to_event(row) for row in rows]
