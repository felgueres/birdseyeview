"""
Vision-Language Model (VLLM) narration for satellite imagery sequences.
Generates flowing narratives describing construction progress over time.
"""

import os
from typing import List, Optional
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image
import time


class VLLMNarrator:
    """Generate narrative descriptions of satellite image sequences using VLLMs."""

    def __init__(self, model: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        """
        Initialize the VLLM narrator.

        Args:
            model: Model identifier (e.g., "gemini-1.5-flash", "gemini-1.5-pro")
            api_key: API key for the model service (if None, reads from GOOGLE_API_KEY env var)
        """
        self.model = model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        from google import genai
        from google.genai import types
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model

        self.narrative_history = []

    def _load_image_as_pil(self, image_path: Path) -> Image.Image:
        """Load image from disk as PIL Image."""
        return Image.open(image_path)

    def _create_prompt(self, frame_indices: List[int], total_frames: int, is_first_batch: bool = False) -> str:
        """Create the prompt for the VLLM."""
        if is_first_batch:
            prompt = f"""Analyze these {len(frame_indices)} satellite images of a datacenter construction project taken chronologically.

Provide a single, concise technical paragraph (5-7 sentences) that:
1. Describes the progression of concrete physical changes visible across the frames (buildings, foundations, electrical infrastructure, grading, roads, HVAC equipment)
2. Notes specific construction milestones reached
3. Estimates overall completion percentage (0-100%) with acknowledgment of uncertainty
4. Identifies the current construction phase

Be technical, factual, and concise. Focus only on observable changes between frames. No section headers, just a flowing paragraph.

Frames {min(frame_indices)}-{max(frame_indices)} of {total_frames}:"""
        else:
            prompt = f"""Continue analyzing frames {min(frame_indices)}-{max(frame_indices)} of {total_frames}.

Previous analysis:
{self.narrative_history[-1] if self.narrative_history else ""}

Add a brief continuation paragraph describing new changes visible in these frames."""

        return prompt

    def narrate_sequence(
        self,
        frame_paths: List[Path],
        batch_size: int = 4,
        delay_between_batches: float = 1.0,
        max_frames: int = 4
    ) -> str:
        """
        Generate a narrative for a sequence of satellite images.

        Args:
            frame_paths: List of paths to image files (should be chronologically sorted)
            batch_size: Number of frames to process in each batch
            delay_between_batches: Seconds to wait between API calls
            max_frames: Maximum number of frames to use (evenly spaced from sequence)

        Returns:
            Complete narrative text describing the sequence
        """
        if not frame_paths:
            return "No frames provided for narration."

        if max_frames and len(frame_paths) > max_frames:
            indices = [int(i * (len(frame_paths) - 1) / (max_frames - 1)) for i in range(max_frames)]
            selected_paths = [frame_paths[i] for i in indices]
            print(f"\n{'='*60}")
            print(f"VLLM Narration")
            print(f"Model: {self.model}")
            print(f"Total frames available: {len(frame_paths)}")
            print(f"Selected frames: {max_frames} (evenly spaced)")
            print(f"Selected indices: {indices}")
            print(f"Batch size: {batch_size}")
            print(f"{'='*60}\n")
            frame_paths = selected_paths
        else:
            print(f"\n{'='*60}")
            print(f"VLLM Narration")
            print(f"Model: {self.model}")
            print(f"Total frames: {len(frame_paths)}")
            print(f"Batch size: {batch_size}")
            print(f"{'='*60}\n")

        full_narrative = []

        for batch_start in range(0, len(frame_paths), batch_size):
            batch_end = min(batch_start + batch_size, len(frame_paths))
            batch_paths = frame_paths[batch_start:batch_end]
            batch_indices = list(range(batch_start, batch_end))

            is_first = batch_start == 0

            print(f"Processing batch: frames {batch_start}-{batch_end-1}")
            print(f"  Loading {len(batch_paths)} images...")

            images = [self._load_image_as_pil(p) for p in batch_paths]

            prompt = self._create_prompt(batch_indices, len(frame_paths), is_first)

            print(f"  Generating narration...")

            try:
                contents = [prompt] + images
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config={"response_modalities": ["TEXT"]}
                )
                narrative_chunk = response.text

                self.narrative_history.append(narrative_chunk)
                full_narrative.append(narrative_chunk)

                print(f"  Generated {len(narrative_chunk)} characters")
                print(f"\n{narrative_chunk}\n")

            except Exception as e:
                error_msg = f"Error generating narration for batch {batch_start}-{batch_end}: {e}"
                print(f"  {error_msg}")
                full_narrative.append(error_msg)

            if batch_end < len(frame_paths):
                print(f"  Waiting {delay_between_batches}s before next batch...")
                time.sleep(delay_between_batches)

        complete_narrative = "\n\n".join(full_narrative)

        print(f"\n{'='*60}")
        print(f"Narration complete!")
        print(f"Total length: {len(complete_narrative)} characters")
        print(f"{'='*60}\n")

        return complete_narrative

    def save_narrative(self, narrative: str, output_path: Path) -> None:
        """Save narrative to a text file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(narrative)
        print(f"Narrative saved to: {output_path}")
