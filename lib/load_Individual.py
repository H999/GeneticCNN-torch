import torch


def load_Individual(name='model.pt'):
    checkpoint = torch.load(name)
    return checkpoint['model'], checkpoint['optimizer'], checkpoint['scheduler']
