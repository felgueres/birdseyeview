# BirdView Overlay UI Usage

## Overview

The new Info Overlay system displays real-time metrics and events directly on your video feed. No extra windows or dependencies needed!

## Features

### 📊 Real-time Metrics Display
- **Frame Count & FPS**: Monitor performance
- **Detection Stats**: Number of detected objects
- **Tracking Stats**: Active tracks, tracked objects
- **Class Breakdown**: Count of objects by class (top 3)
- **Motion Energy**: Optical flow metrics
- **Frame Processing Time**: Pipeline performance in ms
- **Scene Graph Info**: VLM analysis results

### 📝 Event Log
- New object detections
- Track creation events
- High-confidence detections
- Significant motion events
- VLM analysis completion
- Automatically scrolls (last 8 events shown)

## Usage

### Basic Usage with Overlay (Default)
```bash
# Overlay is ON by default
python -m bird.cli --enable-segmentation --enable-tracking
```

### Disable Overlay
```bash
# Use --no-overlay flag
python -m bird.cli --enable-segmentation --enable-tracking --no-overlay
```

### Example Commands

**Segmentation with tracking and overlay:**
```bash
python -m bird.cli --enable-segmentation --enable-tracking --enable-box
```

**Scene graph with overlay:**
```bash
python -m bird.cli --enable-scene-graph --enable-tracking
```

**Full pipeline:**
```bash
python -m bird.cli \
  --enable-segmentation \
  --enable-tracking \
  --enable-scene-graph \
  --enable-box
```

## Customization

### Modify Overlay Appearance

Edit `bird/core/pipeline.py` line 38:

```python
# Change position: 'right', 'left', 'top', 'bottom'
overlay = InfoOverlay(position='right', width=350, alpha=0.75)

# Examples:
overlay = InfoOverlay(position='left', width=300, alpha=0.8)   # Left side
overlay = InfoOverlay(position='bottom', width=100, alpha=0.6) # Bottom bar
```

### Change Colors

Edit `bird/vision/overlay.py`:

```python
# In __init__:
self.bg_color = (20, 20, 20)      # Dark background
self.text_color = (255, 255, 255)  # White text
self.alpha = 0.7                    # Transparency
```

### Add Custom Metrics

Edit `bird/core/pipeline.py` in the metrics section (around line 111):

```python
# Add your custom metric
metrics = {
    'Frame': frame_count,
    'FPS': fps,
    # ... existing metrics ...
    'Your Custom Metric': your_value,  # Add here
}
```

### Add Custom Events

Anywhere in the pipeline loop:

```python
# Add an event when something interesting happens
if some_condition:
    events.append("Your custom event message")
```

## Visual Layout

```
┌─────────────────────────────────────┬──────────────────┐
│                                     │ 🐦 BirdView     │
│                                     │ ─────────────── │
│                                     │ Frame: 1234     │
│     Main Video Feed                 │ FPS: 30.5       │
│                                     │ Detections: 5   │
│     [Computer Vision Output]        │ Tracked: 3      │
│                                     │   person: 2     │
│                                     │   car: 1        │
│                                     │ Motion: 45.2    │
│                                     │ Frame Time: 25ms│
│                                     │                 │
│                                     │ Recent Events   │
│                                     │ [10:23:45] New  │
│                                     │ [10:23:46] High │
│                                     │ [10:23:47] VLM  │
└─────────────────────────────────────┴──────────────────┘
```

## Keyboard Controls

- **Q**: Quit the application
- The overlay is non-interactive (no mouse controls needed)

## Performance Impact

- **Minimal**: ~1-2ms overhead per frame
- **Memory**: ~1MB additional
- The overlay is rendered after all CV operations, so it doesn't affect detection/tracking performance

## Troubleshooting

### Overlay not showing
- Check that `--no-overlay` flag is NOT set
- Verify `enable_overlay=True` in VisionConfig

### Text too small/large
Adjust font scale in `overlay.py`:
```python
cv2.putText(..., cv2.FONT_HERSHEY_SIMPLEX, 0.5, ...)
                                          ^^^
                                       Increase/decrease this
```

### Panel too wide/narrow
Change width parameter:
```python
overlay = InfoOverlay(position='right', width=350, ...)
                                             ^^^
                                          Adjust this
```

## Next Steps

For a web-based dashboard (remote viewing, multiple panels, etc.), see the Flask/Streamlit examples in the architecture section of the main README.

