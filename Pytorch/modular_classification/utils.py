"""Contains the utilities of the pipeline"""
import os
from pathlib import Path
import json
import torch
import model_builder

def save_model(model: torch.nn.Module, target_dir: str,
               model_name: str, hidden_units: int, output_shape: int):
    """
    Saves a Pytorch model
    
    Args:
    -----
        model: A Pytorch model to save
        target_dir: Directory where the model is going to be save
        model_name: a file name for the model
    """
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    model_safe_path = target_dir_path / model_name
    model_safe_path.mkdir(parents=True, exist_ok=True)
    model_name_pth = model_name + ".pth"
    model_safe_name = model_safe_path / model_name_pth
    model_name_json = model_name + ".json"
    model_safe_json = model_safe_path / model_name_json


    print(f"[INFO] saving the model to {model_safe_name}")
    torch.save(model.state_dict(), model_safe_name)
    with open(model_safe_json, "w", encoding="utf-8") as json_file:
        json.dump({"hidden_units": hidden_units, "output_shape": output_shape}, json_file, indent=4)

def load_model(model_path: str) -> torch.nn.Module:
    """
    Loads a Pytorch model and make predictions given a image
    """
    model_path_json = model_path + "/" + os.path.basename(model_path) + ".json"
    with open(model_path_json, encoding="utf-8") as json_file:
        data = json.load(json_file)

    model_path_pth = model_path + "/" + os.path.basename(model_path) + ".pth"
    model = model_builder.TinyVGG(3, data["hidden_units"], data["output_shape"])
    model.load_state_dict(torch.load(model_path_pth, weights_only=True))
    return model
