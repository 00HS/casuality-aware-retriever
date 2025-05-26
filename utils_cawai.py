import os
import random
import numpy as np
import json
import torch

import argparse
import yaml

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Causal Retrieval Model Training")
    parser.add_argument("--config_path", type=str, default="config.yaml")
    parser.add_argument("--model_name", type=str, default = 'bert-base-cased')
    parser.add_argument("--model_run_name", type=str, default = 'cawai_dpr')
    parser.add_argument("--wandb_project", type=str, default="Causal Retrieval")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--random_seed", type=int, default=42)
    return parser.parse_args()

def load_config(config_file='config.yaml'):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def save_checkpoint(model, optimizer, epoch, checkpoint_path):
    checkpoint_file = f"{checkpoint_path}.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': {
            'cause_encoder': model.cause_encoder.state_dict(),
            'effect_encoder': model.effect_encoder.state_dict(),
            'semantic_encoder': model.semantic_encoder.state_dict(),
        },
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_file)
    print(f"Checkpoint saved at {checkpoint_file}")

def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.cause_encoder.load_state_dict(checkpoint['model_state_dict']['cause_encoder'])
        model.effect_encoder.load_state_dict(checkpoint['model_state_dict']['effect_encoder'])
        model.semantic_encoder.load_state_dict(checkpoint['model_state_dict']['semantic_encoder'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from {checkpoint_path} at epoch {checkpoint['epoch'] + 1}")
        return checkpoint['epoch'] + 1
    else:
        print(f"No checkpoint found at {checkpoint_path}.")
        return 0

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data