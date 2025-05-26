import os
import sys

import faiss
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer
import wandb

from model import Encoder, CausalEncoder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils_cawai import set_seed, parse_args, load_config, save_checkpoint, load_checkpoint
from dataloader import parse_data
from loss import negative_log_loss


def main():
    args = parse_args()
    config = load_config(args.config_path)
    set_seed(args.random_seed)

    best_accuracy = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(project=args.wandb_project, name=args.model_run_name, resume="allow")

    cause_encoder = Encoder(args.model_name, device)
    effect_encoder = Encoder(args.model_name, device)
    semantic_encoder = Encoder(args.model_name, device)
    model = CausalEncoder(cause_encoder, effect_encoder, semantic_encoder, device).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    loss_fn = negative_log_loss
    
    train_data_loader = parse_data(config, 'train')
    dev_data_loader = parse_data(config, 'dev')

    
    optimizer = AdamW(model.parameters(),
                      lr=float(args.learning_rate),
                      weight_decay=0.01,
                      eps=1e-8,
                      betas=(0.9, 0.999),
                      correct_bias=False)

   
    total_steps = len(train_data_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    checkpoint_path = f"checkpoint/{args.model_run_name}"
    start_epoch = load_checkpoint(model, optimizer, checkpoint_path)

    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        train_loss = train(model, train_data_loader, tokenizer, loss_fn, optimizer, scheduler, config, device)

        val_accuracy = evaluate(model, dev_data_loader, tokenizer, config, device)

        wandb.log({"Epoch": epoch + 1, "Train Loss": train_loss, "Validation Accuracy": val_accuracy})
        print(f"Train Loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_checkpoint(model, optimizer, epoch, checkpoint_path)
            print(f"New best model saved at epoch {epoch+1} with validation accuracy: {val_accuracy:.4f}")

    print('Training finished!')
    wandb.finish()

    
    test_data_loader = parse_data(config, 'test')
    print("Evaluating on test set...")
    test_accuracy = evaluate(model, test_data_loader, tokenizer, config, device)
    print(f" Test Accuracy: {test_accuracy:.4f}")

def train(model, data_loader, tokenizer, loss_fn, optimizer, scheduler, config, device):
    model.train()
    losses = []

    for batch in tqdm(data_loader, total=len(data_loader)):
        causes = batch['cause']
        effects = batch['effect']

        cause_encoding = tokenizer(causes, truncation=True, padding=True, max_length=config['dataset']['max_length'], return_tensors='pt').to(device)
        effect_encoding = tokenizer(effects, truncation=True, padding=True, max_length=config['dataset']['max_length'], return_tensors='pt').to(device)

        cause_embeddings, effect_embeddings, semantic_cause_embeddings, semantic_effect_embeddings = model(cause_encoding, effect_encoding)

        cause_encoder_loss = loss_fn(cause_embeddings, semantic_effect_embeddings, device)
        effect_encoder_loss = loss_fn(effect_embeddings, semantic_cause_embeddings, device)
        
        semantic_cause_loss = loss_fn(cause_embeddings,semantic_cause_embeddings, device)
        semantic_effect_loss = loss_fn(effect_embeddings, semantic_effect_embeddings, device)
        
        loss = cause_encoder_loss + effect_encoder_loss + semantic_cause_loss + semantic_effect_loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        losses.append(loss.item())
    
    return np.mean(losses)

def evaluate(model, data_loader, tokenizer, config, device):
    model.eval()

    all_causes = []
    all_effects = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader)):
            causes = batch['cause']
            effects = batch['effect']
            all_causes.extend(causes)
            all_effects.extend(effects)

        cause_encoding = tokenizer(all_causes, truncation=True, padding=True, max_length=config['dataset']['max_length'], return_tensors='pt').to(device)
        effect_encoding = tokenizer(all_effects, truncation=True, padding=True, max_length=config['dataset']['max_length'], return_tensors='pt').to(device)

        cause_embeddings, effect_embeddings, _, _ = model(cause_encoding, effect_encoding)

        index_cause = faiss.IndexFlatIP(cause_embeddings.shape[1])
        index_effect = faiss.IndexFlatIP(effect_embeddings.shape[1])
        
        index_cause.add(cause_embeddings.cpu().numpy())
        index_effect.add(effect_embeddings.cpu().numpy())

        correct_cause = 0
        correct_effect = 0

        print("Retrieving with FAISS...")
        for i in tqdm(range(len(all_causes))):
            D, I = index_effect.search(cause_embeddings[i].cpu().numpy().reshape(1, -1), 1)
            if all_effects[I[0][0]] == all_effects[i]:
                correct_cause += 1

        for i in tqdm(range(len(all_effects))):
            D, I = index_cause.search(effect_embeddings[i].cpu().numpy().reshape(1, -1), 1)
            if all_causes[I[0][0]] == all_causes[i]:
                correct_effect += 1
        
        cause_accuracy = correct_cause / len(all_causes)
        effect_accuracy = correct_effect / len(all_effects)
        
        accuracy = (cause_accuracy + effect_accuracy) / 2
        
        return accuracy
    

if __name__ == "__main__":
    main()

