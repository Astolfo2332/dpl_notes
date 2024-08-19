"""Module to test the model with a custom image"""
import argparse
from PIL import Image
import torch
from torchvision import transforms
from utils import load_model

def main():
    """Function to test the model"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, type=str,
                        help="Directory of the model.pth and model.json")
    parser.add_argument("--img_path", required=True, type=str,
                        help="Image to predict path")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model_dir).to(device)
    data_transformer = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    img = Image.open(args.img_path)
    img_transform = data_transformer(img).unsqueeze(0).to(device)
    with torch.inference_mode():
        y_logit = model(img_transform)
    y_preds = torch.softmax(y_logit, 1)
    y_pred = y_preds.argmax(1).item()
    accuracy = y_preds.max().item()
    classes = {0: "pizza", 1: "steak", 2: "Sushi"}
    print(f"It's a {classes[y_pred]} with {accuracy * 100: .2f} accuracy")

if main() == "__main__":
    main()
