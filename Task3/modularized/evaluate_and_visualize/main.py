
import argparse
from utils import load_model
from evaluate import evaluate_model
from visualize import visualize_predictions
from annotate import annotate_video
class_dict = {
    0: "Bipolar",
    1: "SpecimenBag",
    2: "Grasper",
    3: "Irrigator",
    4: "Scissors",
    5: "Hook",
    6: "Clipper"
}
def main():
    # Argument parsing for model path, dataset YAML, video input/output, and CSV (optional)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help="Path to the trained YOLO model weights")
    parser.add_argument('--data-yaml', type=str, default='datasets.yaml', help="Path to the dataset YAML file")
    parser.add_argument('--video-path', type=str, help="Path to the input video for annotation")
    parser.add_argument('--video-output', type=str, default='annotated_output.mp4', help="Path to save the annotated video")
    args = parser.parse_args()

    # Load the YOLO model
    model = load_model(args.model_path)

    # Evaluate model on the test set
    if args.data_yaml:
        print("\nEvaluating on test set...")
        evaluate_model(model, args.data_yaml, split='test')
    
        # Visualize predictions on a few test images (from the test split in dataset.yaml)
        print("\nVisualizing predictions on the test set...")
        visualize_predictions(model, args.data_yaml, class_dict)

    # Optionally process video if a video path is provided
    if args.video_path:
        print(f"\nAnnotating video: {args.video_path}...")
        annotate_video(args.video_path, args.video_output, model)
    
if __name__ == "__main__":
    main()
