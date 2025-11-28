"""
modal run modal.cli.py
"""

from modal import  App, Image
from pathlib import Path
import os
from dotenv import load_dotenv
ROOT = Path(__file__).resolve().parents[1]

load_dotenv(ROOT / ".env")
HF_TOKEN = os.getenv("HF_TOKEN")

# This installs the bird repo as well, commented for speed rn
# .add_local_dir(str(ROOT), remote_path="/root/birdview", copy=True)
# "pip install -e /root/birdview",

image = (
    Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .run_commands([
        f"pip install python-dotenv",
        "pip install huggingface-hub",
        f"hf auth login --token {HF_TOKEN}",
        "pip install torch==2.9.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126",
        "git clone https://github.com/facebookresearch/sam3.git",
        "pip install decord",
        "pip install einops",
        "cd sam3 && pip install -e .\[notebooks,dev\]",
    ])
)

app = App("birdview", image=image)
model = None

@app.function(
    gpu="T4"
)
def run_sam3_text(image_bytes: bytes, text_prompt: str):
    """Segment objects using text prompts (e.g., 'a bag of groceries')"""
    global model
    if model is None:
        from sam3 import build_sam3_image_model
        bpe_path = "/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
        model = build_sam3_image_model(bpe_path=bpe_path, enable_inst_interactivity=False)

    from PIL import Image
    from sam3.model.sam3_image_processor import Sam3Processor
    import io
    import base64

    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # Create processor and run inference with text prompt
    processor = Sam3Processor(model, confidence_threshold=0.5)
    state = processor.set_image(image)
    state = processor.set_text_prompt(text_prompt, state)

    # Convert masks to base64 for easier transmission
    masks_b64 = []
    if "masks" in state and state["masks"] is not None:
        for i, mask in enumerate(state["masks"]):
            mask_np = mask.squeeze().cpu().numpy().astype('uint8') * 255
            mask_img = Image.fromarray(mask_np, mode='L')
            buffer = io.BytesIO()
            mask_img.save(buffer, format='PNG')
            masks_b64.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))

    return {
        "success": True,
        "prompt": text_prompt,
        "num_detections": len(state.get("masks", [])),
        "boxes": state.get("boxes", []).cpu().tolist() if "boxes" in state else [],
        "scores": state.get("scores", []).cpu().tolist() if "scores" in state else [],
        "masks": masks_b64
    }

@app.function(
    gpu="T4"
)
def run_sam3_interactive(image_bytes: bytes, point_coords=None, point_labels=None, box=None):
    """Interactive segmentation using points or boxes (SAM-like interaction)"""
    global model
    if model is None:
        from sam3 import build_sam3_image_model
        bpe_path = "/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
        model = build_sam3_image_model(bpe_path=bpe_path, enable_inst_interactivity=True)

    from PIL import Image
    from sam3.model.sam3_image_processor import Sam3Processor
    import io
    import numpy as np
    import base64

    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # Create processor and set image
    processor = Sam3Processor(model)
    state = processor.set_image(image)

    # Use point-based prompting via predict_inst
    if point_coords is not None:
        input_points = np.array(point_coords)
        input_labels = np.array(point_labels) if point_labels is not None else np.ones(len(point_coords))

        masks, scores, logits = model.predict_inst(
            state,
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )

        # Convert masks to base64
        masks_b64 = []
        for mask in masks:
            mask_np = mask.astype('uint8') * 255
            mask_img = Image.fromarray(mask_np, mode='L')
            buffer = io.BytesIO()
            mask_img.save(buffer, format='PNG')
            masks_b64.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))

        return {
            "success": True,
            "num_masks": len(masks),
            "scores": scores.tolist(),
            "masks": masks_b64
        }

    return {"success": False, "error": "No prompts provided"}

@app.local_entrypoint()
def main():
    """Demo showing different ways to use SAM3"""
    example_image = Path.joinpath(ROOT, 'sam3/assets/images/groceries.jpg')
    with open(example_image, 'rb') as f:
        image_bytes = f.read()

    # Test 1: Text-based segmentation (detect all objects matching text)
    print("\n=== Text-based Segmentation ===")
    results = run_sam3_text.remote(image_bytes=image_bytes, text_prompt="groceries")
    print(f"Found {results['num_detections']} objects")
    print(f"Scores: {results['scores']}")
    print(f"Boxes: {results['boxes']}")

    # Test 2: Interactive segmentation with point prompts
    print("\n=== Interactive Point-based Segmentation ===")
    # Click in the center of the image
    from PIL import Image
    import io
    img = Image.open(io.BytesIO(image_bytes))
    w, h = img.size
    results = run_sam3_interactive.remote(
        image_bytes=image_bytes,
        point_coords=[[w // 2, h // 2]],  # center point
        point_labels=[1]  # positive point
    )
    print(f"Generated {results['num_masks']} masks")
    print(f"Scores: {results['scores']}")