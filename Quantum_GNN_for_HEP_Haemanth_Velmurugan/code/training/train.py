from tqdm import tqdm
import torch
from utils import plot_auc, read_configurations
import numpy as np

config = read_configurations('./config.json')
batch_size = config['BATCH SIZE']


# TODO: Save best model and metrics


def train_model(model, optimizer, lossFn, epochs, lr, train_dataloader, val_dataloader):

    history = {'train_loss': [], 'val_loss': [],
               'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        train_loss, train_acc = run_model(
            model, epoch, train_dataloader, lossFn, optimizer)
        val_loss, val_acc = run_model(
            model, epoch, val_dataloader, lossFn, optimizer, train=False)
        print()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

    return history


def run_model(model, epoch, loader, lossFn, optimizer, train=True):
    if train:
        model.train()
    else:
        model.eval()

    loss = 0
    net_loss = 0
    correct = 0

    for batch_idx, data in (tqdm(enumerate(loader)) if train else enumerate(loader)):

        target = data.y

        # This will zero out the gradients for this batch.
        optimizer.zero_grad()

        # Run the model on the train data
        output = model(data.x, data.edge_index.type(torch.int64), data.batch)

        target = target.unsqueeze(1).float()

        # Calculate the loss
        loss = lossFn(output, target)
        net_loss += loss.data * batch_size

        if train:
            # dloss/dx for every Variable
            loss.backward()

            # to do a one-step update on our parameter.
            optimizer.step()

        pred = (output > 0).float()
        # since we are working with logits and not probabilities (sigmoid is applied while computing loss), we consider 0 as threshold
        correct += (pred == target).sum()

    acc = correct / len(loader.dataset)
    net_loss /= len(loader.dataset)

    if train:
        print('Train', end=" ")
    else:
        print("Val", end=" ")

    # Print out the loss
    print('Epoch: {} \tLoss: {:.6f}, Accuracy: {}/{} ({:.0f}%)'.format(
          epoch, net_loss, correct, len(loader.dataset),
          100. * acc))

    return net_loss, acc


def test_eval(model, test_dataloader):

    preds = []
    labels = []
    acc = 0
    for data in test_dataloader:

        target = data.y
        labels.append(target.detach().cpu().numpy())

        output = model(data.x, data.edge_index.type(torch.int64), data.batch)
        preds.append(output.detach().cpu().numpy())  # Convert to numpy array
        # probs = Sigmoid()(output).detach().cpu().numpy()  # Convert to numpy array
        # preds.append(copy.deepcopy(output))

        target = target.unsqueeze(1).float()
        pred_labels = (output > 0).float()
        acc += (pred_labels == target).sum().item()

    acc = acc / len(test_dataloader.dataset)
    print("Test accuracy: ", 100. * acc)

    # Concatenate lists into a single numpy array
    labels = np.concatenate(labels, axis=0)
    # Concatenate lists into a single numpy array
    preds = np.concatenate(preds, axis=0)

    return labels, preds


def auc(model, test_dataloader):
    labels, preds = test_eval(model, test_dataloader)
    plot_auc(labels, preds)
