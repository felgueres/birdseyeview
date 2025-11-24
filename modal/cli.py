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
def run_sam3(image_bytes: bytes, prompt: str):
    global model
    if model is None:
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        import numpy as np
        from PIL import Image
        import io
        
        bpe_path = "/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
        model = build_sam3_image_model(bpe_path=bpe_path, enable_inst_interactivity=True)
    
    from PIL import Image
    import io
    import numpy as np
    from sam3.model.sam3_image_processor import Sam3Processor
    
    image = Image.open(io.BytesIO(image_bytes))
    
    # Create processor and set image
    processor = Sam3Processor(model)
    inference_state = processor.set_image(image)
    
    # Simple test: predict with a center point
    h, w = image.size[1], image.size[0]
    input_point = np.array([[w // 2, h // 2]])
    input_label = np.array([1])
    
    # Get predictions
    masks, scores, logits = model.predict_inst(
        inference_state,
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    
    return {
        "success": True,
        "num_masks": len(masks),
        "mask_shape": masks.shape,
        "scores": scores.tolist(),
        "prompt": prompt
    }

@app.local_entrypoint()
def main():
    example_image = Path.joinpath(ROOT, 'sam3/assets/images/groceries.jpg')
    with open(example_image, 'rb') as f:
        image_bytes = f.read()
    results = run_sam3.remote(image_bytes=image_bytes, prompt="a bag of groceries")
    print(results)