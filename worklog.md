11/22/2025
Spent the morning getting an NVIDIA Orin to load SAM3. After 2 hrs of battling with pytorch, turns out the model is actually 3GB in FP16 which everything included for inference, it maxes out the formidable yet unusable 8GB RAM in the nano  
Best is to pay 9 bucks for google collab and run on an L4 GPU, just works  
SAM is impressive. Few notes:  
- You can "click" to indicate to model what you want to segment. Either positively to reinforce or negatively.
- You can propagate forward in video which means that you can get a mask and track that object in the video
- Implications of this is to perform operations on the mask, eg. delete the object, swap it, enhance it, etc

11/21/2025
- Requested access for SAM3
- Add depth estimator model: [Depth Anything](https://github.com/DepthAnything/Depth-Anything-V2)

11/19/2025
Found a list of pedestrian datasets here: https://gitlab.tu-clausthal.de/pka20/Trajectory-Prediction-Pedestrian/-/tree/master
WOMD-Reasoning [ICML 2025](https://github.com/yhli123/WOMD-Reasoning). Language dataset of interaction and driving intention reasoning, builds off of Waymo's datasets (WOMD)
- It's interesting that instead of using as VLM, they translated motion data into plain language to feed the model. Prompts included
- A large-scale benchmark dataset for event recognition in surveillance video
- https://viratdata.org - aerial and ground 
- Aerials are public but labeling not available

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