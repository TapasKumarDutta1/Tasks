from ultralytics import YOLO

def evaluate_model(model: YOLO, data_yaml: str, split: str = 'test'):
    """Evaluate the YOLO model on a specified dataset split."""
    metrics = model.val(data=data_yaml, split=split)
    return metrics
