import torch

from tqdm.auto import tqdm


def train(model, device, train_loader, optimizer, loss_func, epoch):
    model.to(device)
    model.train()
    pbar = tqdm(train_loader, leave=False, desc=f"Train Epoch {epoch}")
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        model.zero_grad()
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        pbar.set_postfix({'loss': '{:.8f}'.format(loss.item())})


def test(model, device, test_loader, loss_func, epoch, show=False):
    model.eval()
    test_loss = 0
    correct = 0
    pbar = tqdm(test_loader, leave=False, desc=f"Test Epoch {epoch}")
    with torch.no_grad():
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_func(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    if show:
        print('\nTest Epoch {}: Average loss: {:.8f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
            epoch, test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    return 1. * correct / len(test_loader.dataset), test_loss


def train_Individual(model, device, train_loader, test_loader, optimizer, loss_func, scheduler, epochs, show=False):
    for epoch in tqdm(range(epochs), leave=False):
        train(model, device, train_loader, optimizer, loss_func, epoch)
        accuracy, loss = test(model, device, test_loader, loss_func, epoch, show)
        scheduler.step()
    return accuracy, loss
