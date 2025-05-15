import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
import segmentation_models_pytorch as smp

from utils import evaluate_model


class Trainer:
    """
    Class for training segmentation models.
    
    Args:
        model (nn.Module): PyTorch model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        test_loader (DataLoader): Test data loader
        loss_fn: Loss function
        optimizer: Optimizer
        device (str): Device to use for training ('cuda' or 'cpu')
        checkpoint_dir (str): Directory to save model checkpoints
    """
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        loss_fn=None,
        optimizer=None,
        learning_rate=1e-4,
        device='cuda',
        checkpoint_dir='checkpoints'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        # Set up loss function if not provided
        self.loss_fn = loss_fn if loss_fn is not None else smp.losses.DiceLoss(mode="binary")
        
        # Set up optimizer if not provided
        self.optimizer = optimizer if optimizer is not None else optim.Adam(
            model.parameters(), lr=learning_rate
        )
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Move model to device
        self.model.to(device)
        
        # Track best performance
        self.best_dice = 0
        self.best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
    
    def train_epoch(self):
        """
        Train the model for one epoch.
        
        Returns:
            float: Average loss for this epoch
        """
        self.model.train()
        total_loss = 0
        
        # Use tqdm for progress tracking
        with tqdm(self.train_loader, desc="Training", unit="batch") as pbar:
            for images, masks in pbar:
                # Move data to device and reshape
                images = images.to(self.device).view(-1, 1, 128, 128)
                masks = masks.to(self.device).view(-1, 1, 128, 128)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss
                loss = self.loss_fn(outputs, masks)
                
                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
        
        # Calculate average loss
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self):
        """
        Validate the model on validation set.
        
        Returns:
            tuple: (IoU score, Dice coefficient)
        """
        iou, dice = evaluate_model(self.model, self.val_loader, self.device)
        return iou, dice
    
    def test(self):
        """
        Test the model on test set.
        
        Returns:
            tuple: (IoU score, Dice coefficient)
        """
        iou, dice = evaluate_model(self.model, self.test_loader, self.device)
        return iou, dice
    
    def save_checkpoint(self, path):
        """
        Save model checkpoint.
        
        Args:
            path (str): Path to save the checkpoint
        """
        torch.save(self.model.state_dict(), path)
        print(f"Model checkpoint saved to {path}")
    
    def load_checkpoint(self, path):
        """
        Load model checkpoint.
        
        Args:
            path (str): Path to load the checkpoint from
        """
        self.model.load_state_dict(torch.load(path), strict=False)
        print(f"Model checkpoint loaded from {path}")
    
    def train(self, epochs=5):
        """
        Train the model for specified number of epochs.
        
        Args:
            epochs (int): Number of epochs to train
            
        Returns:
            dict: Training history
        """
        history = {
            'train_loss': [],
            'val_iou': [],
            'val_dice': []
        }
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train for one epoch
            avg_loss = self.train_epoch()
            history['train_loss'].append(avg_loss)
            print(f"Training Loss: {avg_loss:.4f}")
            
            # Validate
            val_iou, val_dice = self.validate()
            history['val_iou'].append(val_iou)
            history['val_dice'].append(val_dice)
            print(f"Validation IoU: {val_iou:.4f}, Dice: {val_dice:.4f}")
            
            # Save best model
            if val_dice > self.best_dice:
                self.best_dice = val_dice
                self.save_checkpoint(self.best_checkpoint_path)
                print(f"New best model saved! Dice: {val_dice:.4f}")
        
        # Load best model and evaluate on test set
        print("\nTraining completed. Loading best model for testing...")
        self.load_checkpoint(self.best_checkpoint_path)
        
        
        return history
