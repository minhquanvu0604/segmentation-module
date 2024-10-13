import torch

def get_optimiser(config, model, txt_logger):
    if config['optimiser']['type'] == 'adam':
        txt_logger.info(f"Adam Optimiser with lr: {config['lr']}")
        return torch.optim.Adam(model.parameters(), lr=config['lr'])

    elif config['optimiser']['type'] == 'adamw':
        txt_logger.info(f"AdamW Optimiser with lr: {config['lr']} and weight decay: {config['optimiser']['weight_decay']}")
        return torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['optimiser']['weight_decay'])

    else:
        raise ValueError(f"Invalid optimiser: {config['optimiser']['type']}")

def get_scheduler(config, optimizer, txt_logger):
    # @TODO: Config scheduler
    if config['lr_scheduler']['type'] == 'cosine_annealing':
        txt_logger.info(f"Cosine Annealing Scheduler with T_max: {config['lr_scheduler']['T_max']}")
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['lr_scheduler']['T_max'])
    
    elif config['lr_scheduler']['type'] == 'step':
        txt_logger.info(f"Step Scheduler with step size: {config['lr_scheduler']['step_size']} and gamma: {config['lr_scheduler']['gamma']}")
        return torch.optim.lr_scheduler.StepLR(optimizer, 
                                            step_size=config['lr_scheduler']['step_size'], 
                                            gamma=config['lr_scheduler']['gamma'])
    
    elif config['lr_scheduler']['type'] == 'exponential':
        txt_logger.info(f"Exponential Scheduler with gamma: {config['lr_scheduler']['gamma']}")
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['lr_scheduler']['gamma'])
    
    else:
        raise ValueError(f"Invalid learning rate scheduler type: {config['lr_scheduler']['type']}")
    
def get_loss(config, txt_logger):
    if config['loss'] == 'cross_entropy':
        txt_logger.info("Loss function: Cross Entropy")
        return torch.nn.CrossEntropyLoss()

    if config['loss'] == 'bce': # Can't implement yet
        txt_logger.info("Loss function: Binary Cross Entropy")
        raise NotImplementedError("Can't implement yet")
        return torch.nn.BCEWithLogitsLoss()

    elif config['loss'] == 'cross_entropy_dice': # Can't implement yet
        import segmentation_models_pytorch
        dice_loss = segmentation_models_pytorch.losses.DiceLoss(mode='binary', from_logits=True, smooth=0.1)
        cross_entropy_loss = torch.nn.CrossEntropyLoss()
        
        def combined_loss(pred, target):
            return 0.7 * dice_loss(pred, target) + 0.3 * cross_entropy_loss(pred, target)
        txt_logger.info("Loss function: 0.3*Cross Entropy + 0.7*Dice Loss")
        
        raise NotImplementedError("Can't implement yet")
        return combined_loss

    else:
        raise ValueError(f"Invalid loss function: {config['loss']}")