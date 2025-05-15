import argparse
import torch
import segmentation_models_pytorch as smp

from data_preparation import prepare_dataloaders
from trainer import Trainer


def get_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Medical Image Segmentation Training')
    
    # Data paths
    parser.add_argument('--images-dir', type=str, required=True,
                        help='Directory containing image volumes')
    parser.add_argument('--masks-dir', type=str, required=True,
                        help='Directory containing segmentation masks')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate for optimizer')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    # Model parameters
    parser.add_argument('--encoder', type=str, default='resnet50',
                        help='Encoder backbone for the U-Net model')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained weights for encoder')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint to continue training')
    
    # Image preprocessing
    parser.add_argument('--img-size', type=int, default=512,
                        help='Target size for image resizing')
    
    # Device settings
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training (cuda or cpu)')
    
    # Output settings
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    
    return parser.parse_args()


def main():
    """
    Main function to run the training pipeline.
    """
    # Parse arguments
    args = get_args()
    
    # Prepare data loaders
    train_loader, val_loader, test_loader = prepare_dataloaders(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        batch_size=args.batch_size,
        target_shape=(args.img_size, args.img_size),
        num_workers=args.num_workers
    )
    
    # Create model
    model = smp.Unet(
        encoder_name=args.encoder,
        encoder_weights="imagenet" if args.pretrained else None,
        in_channels=1,
        classes=1,
    )
    
    # Create loss function
    loss_fn = smp.losses.DiceLoss(mode="binary")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        learning_rate=args.learning_rate,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Start training
    trainer.train(epochs=args.epochs)


if __name__ == "__main__":
    main()
