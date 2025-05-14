import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

from dataset import VolumeSliceDataset
from utils import evaluate_model


def visualize_predictions(model, dataloader, device, num_samples=5):
    """
    Visualize model predictions alongside ground truth masks.
    
    Args:
        model (nn.Module): Trained PyTorch model
        dataloader (DataLoader): DataLoader containing test data
        device (str): Device to run inference on
        num_samples (int): Number of samples to visualize
    """
    model.eval()
    model.to(device)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 5))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    sample_idx = 0
    with torch.no_grad():
        for images, masks in dataloader:
            if sample_idx >= num_samples:
                break
                
            # Get a random slice from the batch
            batch_size = images.size(0)
            slice_idx = np.random.randint(0, images.size(1))
            
            # Get the image and mask
            image = images[0, slice_idx].unsqueeze(0).to(device)
            mask = masks[0, slice_idx].numpy()
            
            # Make prediction
            output = model(image)
            pred_mask = torch.sigmoid(output).cpu().numpy()[0, 0]
            pred_mask_binary = (pred_mask > 0.5).astype(np.uint8)
            
            # Display results
            axes[sample_idx, 0].imshow(image.cpu().numpy()[0, 0], cmap='gray')
            axes[sample_idx, 0].set_title('Input Image')
            axes[sample_idx, 0].axis('off')
            
            axes[sample_idx, 1].imshow(mask[0], cmap='gray')
            axes[sample_idx, 1].set_title('Ground Truth')
            axes[sample_idx, 1].axis('off')
            
            axes[sample_idx, 2].imshow(pred_mask_binary, cmap='gray')
            axes[sample_idx, 2].set_title('Prediction')
            axes[sample_idx, 2].axis('off')
            
            sample_idx += 1
    
    plt.tight_layout()
    plt.savefig('prediction_visualization.png')
    plt.show()


def get_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Medical Image Segmentation Inference')
    
    # Data paths
    parser.add_argument('--images-dir', type=str, required=True,
                        help='Directory containing image volumes')
    parser.add_argument('--masks-dir', type=str, required=True,
                        help='Directory containing segmentation masks')
    parser.add_argument('--test-ids', type=str, nargs='+', required=True,
                        help='List of patient IDs for testing')
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--encoder', type=str, default='resnet50',
                        help='Encoder backbone used in the U-Net model')
    
    # Image preprocessing
    parser.add_argument('--img-size', type=int, default=128,
                        help='Target size for image resizing')
    
    # Device settings
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for inference (cuda or cpu)')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize model predictions')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of samples to visualize')
    
    return parser.parse_args()


def main():
    """
    Main function to run the inference pipeline.
    """
    # Parse arguments
    args = get_args()
    
    # Create test dataset and dataloader
    test_dataset = VolumeSliceDataset(
        args.images_dir, 
        args.masks_dir, 
        args.test_ids, 
        target_shape=(args.img_size, args.img_size)
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=4
    )
    
    # Create model
    model = smp.Unet(
        encoder_name=args.encoder,
        encoder_weights=None,  # No need for pretrained weights during inference
        in_channels=1,
        classes=1,
    )
    
    # Load trained model
    model.load_state_dict(torch.load(args.checkpoint))
    model.to(args.device)
    
    # Evaluate model
    iou, dice = evaluate_model(model, test_loader, args.device)
    print(f"Test Results - IoU: {iou:.4f}, Dice: {dice:.4f}")
    
    # Visualize predictions if requested
    if args.visualize:
        visualize_predictions(model, test_loader, args.device, args.num_samples)


if __name__ == "__main__":
    main()
