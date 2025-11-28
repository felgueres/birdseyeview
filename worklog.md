11/28/2025
the input is an rgb MxNx3 frame and the output is often a MxNx1 binary mask  
the repo most be built in a composable way such that you can apply a sequence of transforms on the frames. I think building that from first principles is important, will spend next few days learning 

11/26/2025
Temporal segmentation is done with calculating the distance between frames. 
You can either compute features classically or use an embedding model.

11/25/2025
Added UI and wired up with remote inference.
Inference latency seems to heavily depend on image size.
The dopamine from working with images is incredible, very different than text.
Considering buying a few cameras and switch box to attempt a Tesla-like, 3d representation of my home. But I think it's best to keep constraints for now otherwise it'll be noise in the learning process.

11/24/2025
Got modal working. I can now call SAM3 on any sized GPU from the repo
This makes it easy to run the model in general but also ready to serve on an app 

11/23/2025
Read "In the blink of an eye" which are the memoirs of a film editor
In the 90s, the industry went from mechanical to digital when computer memory got really cheap
Over night the industry went from Linear Video Editing to Random Access Editing, fully digital
The implications are major and obvious but also subtle and negative
In an extreme exercise to imagine the future, he thinks of a crewless, individual creator that is able to will scenes from his mind into existance and how that creative process fares out. This is the future that has already arrived.

11/22/2025
Spent the morning getting an NVIDIA Orin to load SAM3. After 2 hrs of battling with pytorch, turns out the model is actually 3GB in FP16 which everything included for inference, it maxes out the formidable yet unusable 8GB RAM 
Best is to pay 9 bucks for google collab and run on an L4 GPU
Ran SAM on collab, it's impressive:
- You can "click" to indicate to model what you want to segment. Either click "positively" or "negatively".
- You can propagate forward in video which means that you can get a mask and track that object in the video
- Implications of this is to perform operations on the mask, eg. delete the object, swap it, enhance it, etc

11/21/2025
- Requested access for SAM3
- Added depth estimator model: [Depth Anything](https://github.com/DepthAnything/Depth-Anything-V2)

11/19/2025
Found a list of pedestrian datasets here: https://gitlab.tu-clausthal.de/pka20/Trajectory-Prediction-Pedestrian/-/tree/master
WOMD-Reasoning [ICML 2025](https://github.com/yhli123/WOMD-Reasoning). Language dataset of interaction and driving intention reasoning, builds off of Waymo's datasets (WOMD)
- It's interesting that instead of using as VLM, they translated motion data into plain language to feed the model. Prompts included
- A large-scale benchmark dataset for event recognition in surveillance video
- https://viratdata.org - aerial and ground 
- Aerials are public but labeling not available

11/18/2025
next: 
- add depth estimation (done)
- add temporal segmentation, emit an event on phase change
- add anomaly detection
- add tracking + forecast (world model)
- investigate how embeddings are used 
- refactor pipeline to a dag, should run in parallel
- get eval dataset, get precision/recall metrics
- Wire up in a UI 

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