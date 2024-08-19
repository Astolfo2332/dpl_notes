"""Module to test the model with a custom image"""
import torch
import argparse
from utils import load_model

def main():
    """Function to test the model"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, type=str,
                        help="Directory of the model.pth and model.json")
    args = parser.parse_args()
    model = load_model(args.model_dir)




if main() == "__main__":
    main()