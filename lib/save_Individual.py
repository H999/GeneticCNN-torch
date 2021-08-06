import torch


def save_Individual(model, optimizer, scheduler, name='model.pt'):
    torch.save({
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
    }, name)
