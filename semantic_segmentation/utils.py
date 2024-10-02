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

def colorstr(*input):
    r"""
    From ultralytics/utils/__init__.py

    Colors a string based on the provided color and style arguments. Utilizes ANSI escape codes.
    See https://en.wikipedia.org/wiki/ANSI_escape_code for more details.

    This function can be called in two ways:
        - colorstr('color', 'style', 'your string')
        - colorstr('your string')

    In the second form, 'blue' and 'bold' will be applied by default.

    Args:
        *input (str | Path): A sequence of strings where the first n-1 strings are color and style arguments,
                      and the last string is the one to be colored.

    Supported Colors and Styles:
        Basic Colors: 'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'
        Bright Colors: 'bright_black', 'bright_red', 'bright_green', 'bright_yellow',
                       'bright_blue', 'bright_magenta', 'bright_cyan', 'bright_white'
        Misc: 'end', 'bold', 'underline'

    Returns:
        (str): The input string wrapped with ANSI escape codes for the specified color and style.

    Examples:
        >>> colorstr("blue", "bold", "hello world")
        >>> "\033[34m\033[1mhello world\033[0m"
    """
    *args, string = input if len(input) > 1 else ("blue", "bold", input[0])  # color arguments, string
    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]
