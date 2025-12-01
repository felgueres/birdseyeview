import numpy as np
import torch
from typing import List, Tuple, Optional, Dict
import cv2
from pathlib import Path


class TemporalSegmenter:
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        similarity_threshold: float = 0.99,
        min_segment_length: int = 5
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.similarity_threshold = similarity_threshold
        self.min_segment_length = min_segment_length
        self.model_name = model_name

        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        from transformers import CLIPModel, CLIPProcessor

        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def extract_frame_embeddings(
        self,
        frames: List[np.ndarray],
        batch_size: int = 32
    ) -> np.ndarray:
        embeddings = []

        with torch.no_grad():
            for i in range(0, len(frames), batch_size):
                batch = frames[i:i + batch_size]

                inputs = self.processor(
                    images=batch,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)

                image_features = self.model.get_image_features(**inputs)

                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                embeddings.append(image_features.cpu().numpy())

        return np.vstack(embeddings)

    def compute_similarity_scores(
        self,
        embeddings: np.ndarray,
        window_size: int = 1
    ) -> np.ndarray:
        similarities = []

        for i in range(len(embeddings) - window_size):
            emb1 = embeddings[i]
            emb2 = embeddings[i + window_size]

            similarity = np.dot(emb1, emb2)
            similarities.append(similarity)

        similarities.append(similarities[-1])

        return np.array(similarities)

    def detect_change_points(
        self,
        similarities: np.ndarray,
        method: str = "threshold"
    ) -> List[int]:
        if method == "threshold":
            change_points = []
            for i, sim in enumerate(similarities):
                if sim < self.similarity_threshold:
                    change_points.append(i)
            return change_points

        elif method == "derivative":
            grad = np.gradient(similarities)

            threshold = np.std(grad) * 2
            change_points = []
            for i, g in enumerate(grad):
                if g < -threshold:
                    change_points.append(i)
            return change_points

        elif method == "adaptive":
            window = 10
            change_points = []

            for i in range(window, len(similarities) - window):
                local_mean = np.mean(similarities[i-window:i+window])
                local_std = np.std(similarities[i-window:i+window])

                if similarities[i] < (local_mean - 2 * local_std):
                    change_points.append(i)

            return change_points

        else:
            raise ValueError(f"Unknown method: {method}")

    def merge_nearby_changes(
        self,
        change_points: List[int],
        min_distance: int = 10
    ) -> List[int]:
        if not change_points:
            return []

        merged = [change_points[0]]

        for cp in change_points[1:]:
            if cp - merged[-1] >= min_distance:
                merged.append(cp)

        return merged

    def create_segments(
        self,
        change_points: List[int],
        total_frames: int
    ) -> List[Tuple[int, int]]:
        if not change_points:
            return [(0, total_frames - 1)]

        segments = []
        start = 0

        for cp in change_points:
            if cp - start >= self.min_segment_length:
                segments.append((start, cp))
                start = cp

        if total_frames - start >= self.min_segment_length:
            segments.append((start, total_frames - 1))
        elif segments:
            segments[-1] = (segments[-1][0], total_frames - 1)
        else:
            segments.append((0, total_frames - 1))

        return segments

    def segment_video(
        self,
        video_path: str,
        method: str = "threshold",
        batch_size: int = 32,
        sample_rate: int = 1
    ) -> Dict:
        frames = self._load_video_frames(video_path, sample_rate)

        embeddings = self.extract_frame_embeddings(frames, batch_size)

        similarities = self.compute_similarity_scores(embeddings)

        change_points = self.detect_change_points(similarities, method)

        change_points = self.merge_nearby_changes(change_points, min_distance=10)

        segments = self.create_segments(change_points, len(frames))

        segments_with_time = []
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        for start_frame, end_frame in segments:
            actual_start = start_frame * sample_rate
            actual_end = end_frame * sample_rate

            segments_with_time.append({
                'start_frame': actual_start,
                'end_frame': actual_end,
                'start_time': actual_start / fps,
                'end_time': actual_end / fps,
                'duration': (actual_end - actual_start) / fps
            })

        return {
            'segments': segments_with_time,
            'change_points': [cp * sample_rate for cp in change_points],
            'total_segments': len(segments_with_time),
            'fps': fps,
            'total_frames': len(frames) * sample_rate,
            'sample_rate': sample_rate,
            'similarities': similarities.tolist()
        }

    def _load_video_frames(
        self,
        video_path: str,
        sample_rate: int = 1
    ) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

            frame_idx += 1

        cap.release()
        return frames

    def analyze_frame_batch(
        self,
        frames: List[np.ndarray]
    ) -> Dict:
        embeddings = self.extract_frame_embeddings(frames, batch_size=len(frames))

        similarities = self.compute_similarity_scores(embeddings)

        return {
            'embeddings': embeddings,
            'similarities': similarities.tolist(),
            'mean_similarity': float(np.mean(similarities)),
            'std_similarity': float(np.std(similarities))
        }
