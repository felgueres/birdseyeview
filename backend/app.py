import os
import io
import base64
import uuid
from pathlib import Path
from flask import Flask, request, jsonify, send_file, render_template
import numpy as np
from PIL import Image
import cv2

from bird.vision.detector import ObjectDetector
from bird.vision.depth_estimator import DepthEstimator
from bird.config import VisionConfig
import modal

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

UPLOAD_FOLDER = Path('/tmp/birdview_uploads')
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

vision_config = VisionConfig(
    enable_box=True,
    enable_segmentation=True,
    enable_depth=True,
    depth_model_size='small'
)
detector = ObjectDetector(vision_config=vision_config)
depth_estimator = DepthEstimator(model_size='small')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_image_path(image_id):
    """Get full path for an image ID"""
    for ext in ALLOWED_EXTENSIONS:
        path = UPLOAD_FOLDER / f"{image_id}.{ext}"
        if path.exists():
            return path
    return None

@app.route('/')
def index():
    """Serve the main viewer interface"""
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_image():
    """Upload an image and return an image_id"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        image_id = str(uuid.uuid4())
        ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{image_id}.{ext}"
        filepath = UPLOAD_FOLDER / filename
        file.save(filepath)
        img = Image.open(filepath)
        width, height = img.size
        return jsonify({
            'image_id': image_id,
            'filename': file.filename,
            'width': width,
            'height': height
        })
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/image/<image_id>')
def get_image(image_id):
    """Retrieve an uploaded image"""
    filepath = get_image_path(image_id)
    if filepath is None:
        return jsonify({'error': 'Image not found'}), 404

    return send_file(filepath, mimetype='image/jpeg')

@app.route('/api/segment/interactive', methods=['POST'])
def segment_interactive():
    """
    Interactive SAM3 segmentation using points or boxes
    Body: {
        image_id: str,
        point_coords: [[x, y], ...],
        point_labels: [1, 0, ...],  # 1=positive, 0=negative
        box: [x1, y1, x2, y2]  # optional
    }
    """
    data = request.json
    image_id = data.get('image_id')
    point_coords = data.get('point_coords')
    point_labels = data.get('point_labels')
    box = data.get('box')

    filepath = get_image_path(image_id)
    if filepath is None:
        return jsonify({'error': 'Image not found'}), 404

    with open(filepath, 'rb') as f:
        image_bytes = f.read()

    try:
        run_sam3_interactive = modal.Function.from_name("birdview", "run_sam3_interactive")
        result = run_sam3_interactive.remote(
            image_bytes=image_bytes,
            point_coords=point_coords,
            point_labels=point_labels,
            box=box
        )
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'SAM3 inference failed: {str(e)}'}), 500

@app.route('/api/segment/text', methods=['POST'])
def segment_text():
    """
    Text-based SAM3 segmentation
    Body: {
        image_id: str,
        text_prompt: str
    }
    """
    data = request.json
    image_id = data.get('image_id')
    text_prompt = data.get('text_prompt')

    filepath = get_image_path(image_id)
    if filepath is None:
        return jsonify({'error': 'Image not found'}), 404

    # Read image as bytes
    with open(filepath, 'rb') as f:
        image_bytes = f.read()

    # Call Modal SAM3 function
    try:
        import modal

        run_sam3_text = modal.Function.from_name("birdview", "run_sam3_text")

        result = run_sam3_text.remote(
            image_bytes=image_bytes,
            text_prompt=text_prompt
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'SAM3 inference failed: {str(e)}'}), 500

@app.route('/api/detect', methods=['POST'])
def detect_objects():
    """
    Run object detection on an image
    Body: {
        image_id: str,
        enable_segmentation: bool (optional)
    }
    """
    data = request.json
    image_id = data.get('image_id')
    enable_seg = data.get('enable_segmentation', False)

    filepath = get_image_path(image_id)
    if filepath is None:
        return jsonify({'error': 'Image not found'}), 404

    # Load image with OpenCV
    frame = cv2.imread(str(filepath))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run detection
    detections = detector.detect_objects(frame)

    # Convert numpy arrays to lists for JSON serialization
    result = []
    for det in detections:
        obj = {
            'class': det['class'],
            'confidence': float(det['confidence']),
            'bbox': det['bbox']
        }

        if 'mask' in det and det['mask'] is not None:
            # Encode mask as base64 PNG
            mask = (det['mask'] * 255).astype(np.uint8)
            mask_img = Image.fromarray(mask, mode='L')
            buffer = io.BytesIO()
            mask_img.save(buffer, format='PNG')
            obj['mask'] = base64.b64encode(buffer.getvalue()).decode('utf-8')

        result.append(obj)

    return jsonify({
        'success': True,
        'detections': result,
        'count': len(result)
    })

@app.route('/api/depth', methods=['POST'])
def estimate_depth():
    """
    Estimate depth map for an image
    Body: {
        image_id: str,
        return_visualization: bool (optional)
    }
    """
    data = request.json
    image_id = data.get('image_id')
    return_viz = data.get('return_visualization', True)

    filepath = get_image_path(image_id)
    if filepath is None:
        return jsonify({'error': 'Image not found'}), 404

    # Load image with OpenCV
    frame = cv2.imread(str(filepath))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Estimate depth
    depth_map = depth_estimator.estimate_depth(frame)
    depth_stats = depth_estimator.get_depth_statistics(depth_map)

    result = {
        'success': True,
        'statistics': {
            'mean_depth': float(depth_stats['mean_depth']),
            'median_depth': float(depth_stats['median_depth']),
            'min_depth': float(depth_stats['min_depth']),
            'max_depth': float(depth_stats['max_depth'])
        }
    }

    if return_viz:
        # Create visualization
        depth_viz = depth_estimator.create_depth_overlay(frame, depth_map, alpha=0.6)
        depth_viz = cv2.cvtColor(depth_viz, cv2.COLOR_RGB2BGR)

        # Encode as base64 JPEG
        _, buffer = cv2.imencode('.jpg', depth_viz)
        result['visualization'] = base64.b64encode(buffer).decode('utf-8')

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
