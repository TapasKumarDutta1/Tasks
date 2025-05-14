import cv2
import time
from tqdm import tqdm
import supervision as sv

def annotate_video(input_path: str, output_path: str, model):
    """Annotate a video file frame-by-frame with predictions."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    total_inference_time = 0
    processed_frames = 0

    with tqdm(total=frame_count, desc="Processing video") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            start_time = time.time()
            results = model.predict(frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results).with_nms()

            annotated_frame = frame.copy()
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

            inference_time = time.time() - start_time
            total_inference_time += inference_time
            processed_frames += 1

            out.write(annotated_frame)
            pbar.update(1)

    cap.release()
    out.release()

    avg_fps = processed_frames / total_inference_time if total_inference_time > 0 else 0
    print(f"Annotated video saved to {output_path}")
    print(f"Processed {processed_frames} frames in {total_inference_time:.2f} seconds — Avg FPS: {avg_fps:.2f}")
