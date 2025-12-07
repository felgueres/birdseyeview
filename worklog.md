12/7/2025 There's active research off of Sentinel2.
[SatClip](https://arxiv.org/pdf/2311.17179) seems interesting, it's general
purpose location embeddings which means: text/img in -> embedding out. This is
cool, download randomnly sampled tiles, chunk, embed, store in db, UI to map
world map with tiles and a search box, click or search by text. Sentinel2 is 56K
tiles, but 22K are land.

Parcel tasks: extract parcel boundaries, land cover classification, crop type
classifiers (using a timeseries of S2), veg indicators (NDVI, phenology,
greenup, stress) which infers yield forecasts, change detection, crop
management. Probably the least explored app is emb-based similar parcel finding.

One RGB earth-pass is 22K tiles, 55MB/tile, ~1TB. So a single band is 333GB.
Sentinel-2 has 13 bands, 4.3TB/earth-pass multispectral at 10m/pixel resolution.

Business models: track changes / km2 (eg. ~10K/year for 50km2, so ~200USD/km2)

12/6/2025 Sentinel2 is a free open source satellite data by the EU. Getting
tiles is easy, resolution is 10meters per pixel, commercial sats give you 1 or
even less (insane). You can transform the data in many ways to make it easier to
make inferences, and there's satellite specific models that work well for
segmentation tasks. I think the simplest workflow is tracking change. Let a user
define a parcel and run checks on every satellite pass: farmland, parking lots,
stations, remote areas, construction sites, ports.

Place a polygon, tell me what changed. Hill identified, climbing.

12/1/2025 added sqlite and ability to query over vlm events. i now have this
loop:

1. video frames as input
2. a frame is processed by a composable pipeline in a dag
3. the unit of processing is a Transform, which varies from classic computer
   vision tasks like event extraction, getting embeddings, and sinking the data
   to sqlite
4. search and retrieval of scenes by frame
5. runs from the cli. spending time on the ui is a waste of time

The next step is grabbing an eval and hill climb to make events extraction,
search, the transforms better and faster. Software tooling first, then hardware.

11/30/2025 video is the current dominant form of the internet. there are more
bits in the form of video than text. however video is lower value per bit. so
humans want to consume video despite it being less valuable than text. video
understanding through ai helps compress this high information, low value medium
into a high value one

ideas: unmanned operation and maintenance

11/29/2025 most engineering problems are solved with good tooling. solve tooling
and the solution to the problem follows. video understanding it's the og
streaming problem but unlike llm tokens, every frame goes through heavy
processing instead of being a pass through. thinking about it, might be that
we'll see the streaming libraries coming to llm world as more tools are used in
a loop

11/28/2025 the input is an rgb MxNx3 frame and the output is often a MxNx1
binary mask  
the repo most be built in a composable way such that you can apply a sequence of
transforms on the frames. I think building that from first principles is
important, will spend next few days learning. consider for running models on the
browser: https://onnxruntime.ai/

feels like im building towards a video processor more than a video generator.
frame understanding. looking around, capcut is wild, it's basically slop on
steroids for video. frivolous beyond belief. energy and ai are the most
important industries in my lifetime. to what capacity does video understanding
helps: distribution (interconnection, transmission), manufacturing backlog, slow
regulatory environment

11/26/2025 Temporal segmentation is done with calculating the distance between
frames. You can either compute features classically or use an embedding model.

11/25/2025 Added UI and wired up with remote inference. Inference latency seems
to heavily depend on image size. The dopamine from working with images is
incredible, very different than text. Considering buying a few cameras and
switch box to attempt a Tesla-like, 3d representation of my home. But I think
it's best to keep constraints for now otherwise it'll be noise in the learning
process.

11/24/2025 Got modal working. I can now call SAM3 on any sized GPU from the repo
This makes it easy to run the model in general but also ready to serve on an app

11/23/2025 Read "In the blink of an eye" which are the memoirs of a film editor
In the 90s, the industry went from mechanical to digital when computer memory
got really cheap Over night the industry went from Linear Video Editing to
Random Access Editing, fully digital The implications are major and obvious but
also subtle and negative In an extreme exercise to imagine the future, he thinks
of a crewless, individual creator that is able to will scenes from his mind into
existance and how that creative process fares out. This is the future that has
already arrived.

11/22/2025 Spent the morning getting an NVIDIA Orin to load SAM3. After 2 hrs of
battling with pytorch, turns out the model is actually 3GB in FP16 which
everything included for inference, it maxes out the formidable yet unusable 8GB
RAM Best is to pay 9 bucks for google collab and run on an L4 GPU Ran SAM on
collab, it's impressive:

- You can "click" to indicate to model what you want to segment. Either click
  "positively" or "negatively".
- You can propagate forward in video which means that you can get a mask and
  track that object in the video
- Implications of this is to perform operations on the mask, eg. delete the
  object, swap it, enhance it, etc

11/21/2025

- Requested access for SAM3
- Added depth estimator model:
  [Depth Anything](https://github.com/DepthAnything/Depth-Anything-V2)

11/19/2025 Found a list of pedestrian datasets here:
https://gitlab.tu-clausthal.de/pka20/Trajectory-Prediction-Pedestrian/-/tree/master
WOMD-Reasoning [ICML 2025](https://github.com/yhli123/WOMD-Reasoning). Language
dataset of interaction and driving intention reasoning, builds off of Waymo's
datasets (WOMD)

- It's interesting that instead of using as VLM, they translated motion data
  into plain language to feed the model. Prompts included
- A large-scale benchmark dataset for event recognition in surveillance video
- https://viratdata.org - aerial and ground
- Aerials are public but labeling not available

11/18/2025 Next ideas: tracking + forecast?, investigate how embeddings are
used, refactor to DAG, get eval dataset, add ui

11/17/2025

- VIRAT. DARPA funded surveillance (https://en.wikipedia.org/wiki/VIRAT)
- AVA. Atomic Visual Actions (Google) (https://research.google.com/ava)

Glossary:

- VCA. Video Content Analysis
- Video tracking. Locating moving object over time using a camera.
- Track. An object followed over time (trajectory)
- Segments. Video segments, often used in training (eg. training on 100K
  segments )
