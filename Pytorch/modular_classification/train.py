"""Script to train a Pytorch model"""
import argparse
import os
import torch
import data_setup
import engine
import model_builder
import utils
from torchvision import transforms

WORKERS = os.cpu_count()

def main():
    """Function to parse and train model with given arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs for training")
    parser.add_argument("--hidden_units", type=int, default=10,
                         help="Number of neurons in the hidden layers")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the model")
    parser.add_argument("--train", type=str, required=True, help="Path to the train set")
    parser.add_argument("--test", type=str, required=True, help="Path to the test set")
    parser.add_argument("--out", type=str, required=True,
                         help="Path of the directory to save the model")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device selected: {device}")
    data_transform = transforms.Compose(
        [transforms.Resize((64, 64)),
        transforms.RandomVerticalFlip(0.5),
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ToTensor()]
    )
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        args.train, args.train, data_transform, args.num_epochs, WORKERS
    )

    model = model_builder.TinyVGG(3, args.hidden_units, len(class_names)).to(device)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    engine.train(model, test_dataloader, train_dataloader,
                loss_fn, optimizer, device, args.num_epochs)
    utils.save_model(model, "models", args.out, args.hidden_units, len(class_names))




if __name__ == "__main__":
    main()
