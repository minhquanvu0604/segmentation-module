import os, sys
top_level_package = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # segmentation-module
sys.path.insert(0, top_level_package)

from datetime import datetime
import shutil
from tqdm import tqdm
import yaml

import numpy as np
import torch

from deeplabv3_apples.config.config import INPUT_SIZE, TRAIN_CONFIG, DATASET_PATH, CSV_PATH
from deeplabv3_apples.utils import configure_logger
from deeplabv3_apples.model import get_model, print_model_info
from deeplabv3_apples.engine import get_optimiser, get_scheduler, get_loss
from deeplabv3_apples.data.dataloader import get_transforms, get_dataloaders
from deeplabv3_apples.utils import SaveBestModel, EarlyStopping, save_model, colorstr
from deeplabv3_apples.validate import validate, pix_acc, save_plots


def train(config):

    save_dir = config['save_dir']
    valid_pred_dir = os.path.join(save_dir, 'valid_preds') # Save validation predictions
    os.makedirs(valid_pred_dir, exist_ok=True)


    # TRAIN SETUP ------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu') # For testing

    txt_logger.info(colorstr("yellow", "bold", f"Device: {device}"))
    txt_logger.info(colorstr("green", "bold", f"Number of GPUs: {torch.cuda.device_count()}"))
    txt_logger.info(colorstr("yellow", "bold", f"Batch size: {config['batch_size']}"))
    txt_logger.info(f"Data splitting CSV: {CSV_PATH}")
    txt_logger.info("Split set: {}".format(config['split']))


    # Model
    model = get_model(config['num_classes'], pretrained=True).to(device)
    print_model_info(txt_logger, model, INPUT_SIZE, device)
    
    # Optimiser
    optimizer = get_optimiser(config, model, txt_logger)

    # Scheduler
    scheduler = get_scheduler(config, optimizer, txt_logger)

    # Loss function
    criterion = get_loss(config, txt_logger)
    
    # Data loaders
    train_transforms, val_transforms = get_transforms(INPUT_SIZE)
    train_loader, val_loader = get_dataloaders(split_set=config['split'],
                                            batch_size=config['batch_size'],
                                            train_transforms=train_transforms,
                                            val_transforms=val_transforms)
    
    # Track best validation performance
    save_best_model = SaveBestModel()

    train_loss, train_pix_acc = [], []
    val_loss, val_pix_acc = [], []

    # Early stopping
    early_stopping = EarlyStopping(patience=5, delta=0.001)

    # TRAINING LOOP ------------------------------------------------
    num_epochs = config['epochs']
    for epoch in range(num_epochs):
        txt_logger.info('------------------------------------------------------------') 
        txt_logger.info(f"EPOCH: {epoch + 1}/{num_epochs}")
        current_lr = scheduler.get_last_lr()[0]
        txt_logger.info(f"Current Learning Rate: {current_lr}")
        txt_logger.info('----')

        # TRAINING PHASE ------------------------------------------------
        model.train()
        running_loss = 0.0
        running_correct, running_labeled = 0, 0
        
        for data, target in tqdm(train_loader, desc='Training'):
            # If you're using GPUs, setting pin_memory=True and using non_blocking=True when moving data to the device can improve data transfer efficiency
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(data)['out']

            loss = criterion(outputs, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            
            running_loss += loss.item()

            # Calculate pixel accuracy
            labeled, correct = pix_acc(target, outputs)
            running_labeled += labeled.sum()
            running_correct += correct

        train_epoch_loss = running_loss / len(train_loader)
        train_epoch_pixacc = 100 * running_correct / (np.spacing(1) + running_labeled)

        train_loss.append(train_epoch_loss)
        train_pix_acc.append(train_epoch_pixacc.cpu()) # @TODO WHy cpu()?

        txt_logger.info(f"Train Loss: {train_epoch_loss:.4f} | Train PixAcc: {train_epoch_pixacc:.2f}%")
        txt_logger.info('------------------------------------------------------------')

        # VALIDATION PHASE ------------------------------------------------
        val_epoch_loss, val_epoch_pixacc = validate(model, 
                                                    val_loader, 
                                                    device, 
                                                    criterion, 
                                                    epoch, 
                                                    valid_pred_dir)
        
        val_loss.append(val_epoch_loss)
        val_pix_acc.append(val_epoch_pixacc.cpu())
        txt_logger.info(f"Validation Loss: {val_epoch_loss:.4f} | Validation PixAcc: {val_epoch_pixacc:.2f}%")
        
        # Save the best model based on validation loss
        save_best_model(val_epoch_loss, epoch, model, save_dir)

        if early_stopping(val_epoch_loss):
            txt_logger.info("Early stopping triggered")
            break

        scheduler.step()
        txt_logger.info(f"Learning rate adjusted to: {scheduler.get_last_lr()[0]}")

    # Save the final model, optimizer, and loss
    save_model(num_epochs, model, optimizer, criterion, save_dir)

    # Save plots for loss and accuracy
    save_plots(train_pix_acc, val_pix_acc, train_loss, val_loss, save_dir)


if __name__ == '__main__':

    # System check 
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at: {DATASET_PATH}")

    # Load train config
    train_config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), TRAIN_CONFIG))
    with open(train_config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Set up directory to save outputs
    start_time = datetime.now()
    save_dir = os.path.join(os.path.dirname(__file__), 'output', start_time.strftime('%Y_%m_%d_%H_%M_%S'))
    os.makedirs(save_dir, exist_ok=True)
    config['save_dir'] = save_dir

    # Loggers
    txt_logger = configure_logger(save_dir)
    txt_logger.info(f"\nConfig file: {train_config_path}")

    # Save config
    with open(os.path.join(save_dir, 'config_used.yaml'), 'w') as file:
        yaml.dump(config, file)
    txt_logger.info(f"Save directory: {save_dir}")


    # ------- START TRAINING ---------
    txt_logger.info("\n" + colorstr("yellow", "bold", config['note']) + "\n")
    txt_logger.info(f"--------- [START] Training started at: {start_time} ---------\n")

    try:
        train(config)
    except Exception as e:
        txt_logger.error(f"Training failed with error: {e}")
        shutil.rmtree(save_dir, ignore_errors=True)
        raise e
    
    end_time = datetime.now()
    txt_logger.info(f"--------- [COMPLETE] Training finished at: {end_time} ---------\n")


    print("[COMPLETE] Training time:")
    print(f"Training started at: {start_time}")
    print(f"Training finished at: {end_time}")
    total_time = end_time - start_time
    print(f"Total training time: {total_time}")
