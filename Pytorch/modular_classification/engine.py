"""Module that can train and test the Models"""
from typing import Dict, List, Tuple
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm.auto import tqdm

def train_func(model: nn.Module, data: DataLoader, loss_fn:nn.Module,
               optimizer: torch.optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    """
    Trains a model for a single epoch

    Args
    ----
        model: A Pytorch model
        data: a DataLoader object with the train data
        loss_fn: loss function to minimized
        optimizer: A Pytorch optimizer
        device: A target device to perform the operations ("cuda" or "cpu")

    Returns
    ------
        A tuple with the loss and accuracy of the training epoch like
        (train_loss, train_acc)
    """
    model.eval()
    train_loss, train_acc = 0, 0
    for _ , (x, y) in enumerate(data):
        x, y = x.to(device), y.to(device)
        y_logit = model(x)
        loss = loss_fn(y_logit, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred = torch.softmax(y_logit, 1).argmax(1)
        train_acc += (y_pred == y).sum().item() / len (y_pred)
        #We can use another function if we want
    train_loss = train_loss / len(data)
    train_acc = train_acc / len(data)
    return train_loss, train_acc

def test_func(model: nn.Module, data: DataLoader,
            loss_fn: nn.Module, device: torch.device) -> Tuple[float, float]:
    """
    Test a model for a single epoch

    Args
    ----
        model: A Pytorch model
        data: a DataLoader object with the train data
        loss_fn: loss function to minimized
        device: A target device to perform the operations ("cuda" or "cpu")

    Returns
    ------
        A tuple with the loss and accuracy of the testing like
        (test_loss, test_acc)
    """
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for _, (x, y) in enumerate(data):
            x, y = x.to(device), y.to(device)
            y_logits = model(x)
            loss = loss_fn(y_logits, y)
            test_loss += loss.item()
            y_pred = y_logits.argmax(1)
            test_acc += (y_pred == y).sum().item() / len (y_pred)
    test_loss = test_loss / len(data)
    test_acc = test_acc / len(data)
    return test_loss, test_acc

def train(model: nn.Module, test_data: DataLoader, train_data: DataLoader, loss_fn:nn.Module,
        optimizer: torch.optim.Optimizer, device: torch.device, epochs: int) -> Dict[str, List]:
    """
    Trains and test a Pytorch model

    Args:
    -----
        model: A Pytorch model
        train_data: a DataLoader object with the train data
        test_data: a DataLoader object with the train data
        loss_fn: loss function to minimized
        optimizer: A Pytorch optimizer
        device: A target device to perform the operations ("cuda" or "cpu")
        epochs: A integre with the number of epochs that the model will be train
    Returns:
    --------
        A dictionary with the train and test loss and accuracy for every epoch
        in the form of 
        {"train_loss": [...],
        "train_acc": [...],
        "test_loss": [...],
        "test_acc": [...]}
    """
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_func(
            model,train_data,loss_fn,optimizer, device)
        test_loss, test_acc = test_func(
            model, test_data, loss_fn, device)
        print(
            f"Epoch {epoch+1} |"
            f"train_loss :{train_loss: .4f} |"
            f"train_acc :{train_acc: .4f} |"
            f"test_loss :{test_loss: .4f} |"
            f"test_acc :{test_acc: .4f} "
        )
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
    return results
