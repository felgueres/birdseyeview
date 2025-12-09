import cv2
from typing import Optional

def frames_to_video(
    self,
    output_path: Optional[str] = None,
    fps: int = 2,
    add_labels: bool = True
):
    """
    Create video from frames.

    Args:
        output_path: Output video path (default: timeseries.mp4)
        fps: Frames per second (2 = slow, 10 = fast)
        add_labels: Add date labels to frames
    """
    if output_path is None:
        output_path = str(self.output_dir / "timeseries.mp4")

    frame_files = sorted(self.output_dir.glob("frame_*.jpg"))
    if not frame_files:
        print("No frames found. Run download_timeseries() first.")
        return

    first_frame = cv2.imread(str(frame_files[0]))
    h, w = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    import json
    meta_path = self.output_dir / "metadata.json"
    metadata = {}
    if meta_path.exists():
        with open(meta_path) as f:
            data = json.load(f)
            metadata = {f['frame_idx']: f for f in data['frames']}

    for i, frame_file in enumerate(frame_files):
        frame = cv2.imread(str(frame_file))

        if add_labels and i in metadata:
            date = metadata[i]['date']
            cloud = metadata[i]['cloud_cover']

            cv2.putText(
                frame,
                f"{date}  |  cloud: {cloud:.1f}%",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            cv2.putText(
                frame,
                f"{date}  |  cloud: {cloud:.1f}%",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

        out.write(frame)

    out.release()
    print(f"Video saved: {output_path}")
    print(f"Duration: {len(frame_files)/fps:.1f}s @ {fps} fps")