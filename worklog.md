11/18/2025
- Up next: 
    - get eval dataset, get precision/recall metrics
    - add temporal segmentation, emit an event on phase change
    - add depth estimation
    - add anomaly detection
    - add tracking + forecast (world model)
    - investigate what embeddings are used for here 
    - refactor pipeline to a dag, should run in parallel

- I would feel proud to build reasoning for video. Works well with my story at Harvey
- "YOLO / SAM / trackers run at 30–90 FPS"
- Current pipeline runs at 70 ms / frame, ~8 FPS
- project names: action, reasonable

11/17/2025

Datasets:
- VIRAT. DARPA funded surveillance (https://en.wikipedia.org/wiki/VIRAT)
- AVA. Atomic Visual Actions (Google) (https://research.google.com/ava)

Glossary:
VCA. Video Content Analysis
Video tracking. Locating moving object over time using a camera.
Track. An object followed over time (trajectory)