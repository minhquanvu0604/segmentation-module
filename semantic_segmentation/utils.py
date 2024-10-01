import os
import logging
import torch
import matplotlib.pyplot as plt


def configure_logger(path):
    logger = logging.getLogger('TRAINING LOGGER')
    logger.setLevel(logging.DEBUG) 

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # File Handler
    log_filename = 'training_log'
    file_handler = logging.FileHandler(f'{path}/{log_filename}.txt', mode='a')
    logger.addHandler(file_handler)
    return logger

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous best validation loss, then
    save the model state.
    """
    def __init__(self, best_valid_loss=float('inf')):
        """
        Initialize the SaveBestModel class.

        Parameters:
            best_valid_loss (float): The initial best validation loss.
                                    Set to infinity to ensure any validation loss
                                    will be better in the first epoch.
        """
        self.best_valid_loss = best_valid_loss
        
    def __call__(self, current_valid_loss, epoch, model, save_dir, model_name='model'):
        """
        Saves the model if the current validation loss is less than the best recorded loss.

        Parameters:
            current_valid_loss (float): The validation loss from the current epoch.
            epoch (int): The current epoch number.
            model (torch.nn.Module): The model instance being trained.
            save_dir (str): Directory where the model should be saved.
            model_name (str): Name prefix for the saved model file.
        """
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"Saving best model for epoch {epoch + 1}\n")
            
            # Save model state
            save_path = os.path.join(save_dir, f'best_{model_name}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'best_valid_loss': self.best_valid_loss
            }, save_path)

def save_model(epochs, model, optimizer, criterion, save_dir, model_name='model'):
    """
    Function to save the trained model to disk.

    Parameters:
        epochs (int): Total number of epochs the model was trained for.
        model (torch.nn.Module): The trained model.
        optimizer (torch.optim.Optimizer): The optimizer used during training.
        criterion (torch.nn.Module): The loss function used during training.
        save_dir (str): The directory where the model will be saved.
        model_name (str): The name to save the model under.
    """
    save_path = os.path.join(save_dir, f'{model_name}.pth')
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
    }, save_path)
    print(f"Model saved to {save_path}")


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
