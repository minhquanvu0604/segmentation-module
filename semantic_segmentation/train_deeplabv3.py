import os, sys
top_level_package = os.path.abspath(os.path.join(os.path.dirname(__file__))) # semantic_segmentation
sys.path.append(top_level_package)

from datetime import datetime
from tqdm import tqdm

import torch

from model import get_model
from data.dataloader import get_transforms, get_dataloaders


def train(config):

    # Save directory
    now = datetime.now()
    subdir = now.strftime('%Y_%m_%d_%H_%M_%S')
    save_directory = os.path.join(config['path_to_output'], subdir)
    os.makedirs(save_directory, exist_ok=True)
    os.makedirs(os.path.join(save_directory, 'valid_preds'), exist_ok=True)

    # Txt logger


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model(config['num_classes'], pretrained=True).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = torch.nn.CrossEntropyLoss()

    train_transforms, val_transforms = get_transforms()
    train_loader, val_loader = get_dataloaders(batch_size=config['batch_size'],
                                               train_transforms=train_transforms,
                                               val_transforms=val_transforms)
    
    # Track best validation performance
    # save_best_model = SaveBestModel()

    train_loss, train_pix_acc = [], []
    val_loss, val_pix_acc = [], []

    num_epochs = config['epochs']

    for epoch in range(num_epochs):
        print(f"EPOCH: {epoch + 1}/{num_epochs}")

        # TRAINING PHASE
        model.train()
        running_loss = 0.0
        running_correct, running_labeled = 0, 0
        
        for data, target in tqdm(train_loader, desc='Training'):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            outputs = model(data)['out']
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            # Calculate pixel accuracy
            labeled, correct = pix_acc(target, outputs, config['num_classes'])
            running_labeled += labeled.sum()
            running_correct += correct

        train_epoch_loss = running_loss / len(train_loader)
        train_epoch_pixacc = 100 * running_correct / (np.spacing(1) + running_labeled)

        train_loss.append(train_epoch_loss)
        train_pix_acc.append(train_epoch_pixacc.cpu())

        print(f"Train Loss: {train_epoch_loss:.4f} | Train PixAcc: {train_epoch_pixacc:.2f}%")

        # VALIDATION PHASE
        val_epoch_loss, val_epoch_pixacc = validate(
            model, val_loader, device, criterion,
            config['classes_to_train'], config['label_colors_list'],
            epoch, config['all_classes'], os.path.join(save_directory, 'valid_preds')
        )
        
        val_loss.append(val_epoch_loss)
        val_pix_acc.append(val_epoch_pixacc.cpu())
        
        print(f"Validation Loss: {val_epoch_loss:.4f} | Validation PixAcc: {val_epoch_pixacc:.2f}%")
        
        # Save the best model based on validation loss
        save_best_model(val_epoch_loss, epoch, model, save_directory)

    # Save the final model, optimizer, and loss
    save_model(num_epochs, model, optimizer, criterion, save_directory)

    # Save plots for loss and accuracy
    save_plots(train_pix_acc, val_pix_acc, train_loss, val_loss, save_directory)

    print('TRAINING COMPLETE')

