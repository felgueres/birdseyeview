11/19/2025
Found a list of pedestrian datasets here: https://gitlab.tu-clausthal.de/pka20/Trajectory-Prediction-Pedestrian/-/tree/master
WOMD-Reasoning [ICML 2025](https://github.com/yhli123/WOMD-Reasoning). Language dataset of interaction and driving intention reasoning, builds off of Waymo's datasets (WOMD)
- It's interesting that instead of using as VLM, they translated motion data into plain language to feed the model. Prompts included

11/18/2025
next: 
- get eval dataset, get precision/recall metrics
- add temporal segmentation, emit an event on phase change
- add depth estimation
- add anomaly detection
- add tracking + forecast (world model)
- investigate how embeddings are used 
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