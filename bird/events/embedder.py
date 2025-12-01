import numpy as np
import torch
from typing import Optional, List, Dict, Any


class EventEmbedder:
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        from transformers import CLIPModel, CLIPTokenizer
        import warnings

        warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')

        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def embed_event(self, event: Dict[str, Any]) -> np.ndarray:
        text = self._event_to_text(event)
        return self.embed_text(text)

    def embed_events(self, events: List[Dict[str, Any]]) -> np.ndarray:
        texts = [self._event_to_text(event) for event in events]
        return self.embed_texts(texts)

    def embed_text(self, text: str) -> np.ndarray:
        with torch.no_grad():
            inputs = self.tokenizer(
                [text],
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features.cpu().numpy()[0].astype(np.float32)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features.cpu().numpy().astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_text(query)

    def _event_to_text(self, event: Dict[str, Any]) -> str:
        parts = []

        event_type = event.get('type', 'unknown')
        parts.append(event_type.replace('_', ' '))

        meta = event.get('meta', {})

        if 'description' in meta:
            parts.append(meta['description'])
        else:
            if 'class' in meta:
                parts.append(f"{meta['class']}")
            if 'object_class' in meta:
                parts.append(f"interacting with {meta['object_class']}")

            for key, value in meta.items():
                if key not in ['class', 'object_class', 'position', 'distance', 'duration']:
                    if isinstance(value, str):
                        parts.append(value)

        return " ".join(parts)
