from ultralytics import YOLO

def load_model(model_path: str) -> YOLO:
    """Load a trained YOLO model from disk."""
    return YOLO(model_path)
