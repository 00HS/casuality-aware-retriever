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

wandb.login()

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

        train_loss = train(model, train_data_loader,loss_fn, optimizer, scheduler, device)

        val_accuracy = evaluate(model, dev_data_loader, device)

        wandb.log({"Epoch": epoch + 1, "Train Loss": train_loss, "Validation Accuracy": val_accuracy})
        print(f"Train Loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_checkpoint(model, optimizer, epoch, checkpoint_path)
            print(f"New best model saved at epoch {epoch+1} with validation accuracy: {val_accuracy:.4f}")

    print('Training finished!')
    wandb.finish()
    
    test_data_loader = parse_data(config, 'test')
    test_accuracy = evaluate(model, test_data_loader, device)
    print(f"Test Loss: {test_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

def train(model, data_loader, loss_fn, optimizer, scheduler, device):
    model.train()
    losses = []

    for batch in tqdm(data_loader, total=len(data_loader), desc="Training"):
        causes = batch['cause']
        effects = batch['effect']

        with torch.set_grad_enabled(True):
            cause_features = model.cause_encoder.tokenize(causes)
            effect_features = model.effect_encoder.tokenize(effects)
            semantic_cause_features = model.semantic_encoder.tokenize(causes)
            semantic_effect_features = model.semantic_encoder.tokenize(effects)
            
            cause_features = {k: v.to(device) for k, v in cause_features.items()}
            effect_features = {k: v.to(device) for k, v in effect_features.items()}
            semantic_cause_features = {k: v.to(device) for k, v in semantic_cause_features.items()}
            semantic_effect_features = {k: v.to(device) for k, v in semantic_effect_features.items()}

            cause_embeddings = model.cause_encoder(cause_features)['sentence_embedding']
            effect_embeddings = model.effect_encoder(effect_features)['sentence_embedding']

            semantic_cause_embeddings = model.semantic_encoder(semantic_cause_features)['sentence_embedding']
            semantic_effect_embeddings = model.semantic_encoder(semantic_effect_features)['sentence_embedding']
            
            cause_encoder_loss  = loss_fn(cause_embeddings, semantic_effect_embeddings, device)
            effect_encoder_loss = loss_fn(effect_embeddings, semantic_cause_embeddings, device)

            semantic_cause_loss = loss_fn(cause_embeddings, semantic_cause_embeddings, device)
            semantic_effect_loss = loss_fn(effect_embeddings, semantic_effect_embeddings, device)
            
            loss = 1*(semantic_cause_loss + semantic_effect_loss) + cause_encoder_loss + effect_encoder_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())

    return np.mean(losses)

def evaluate(model, data_loader, device):
    model.eval()
    all_causes = []
    all_effects = []
    all_cause_embeddings = []
    all_effect_embeddings = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader), desc="embedding"):
            causes = batch['cause']
            effects = batch['effect']
            
            cause_features = model.cause_encoder.tokenize(causes)
            effect_features = model.effect_encoder.tokenize(effects)
            
            cause_features = {k: v.to(device) for k, v in cause_features.items()}
            effect_features = {k: v.to(device) for k, v in effect_features.items()}
            
            cause_emb = model.cause_encoder(cause_features)['sentence_embedding']
            effect_emb = model.effect_encoder(effect_features)['sentence_embedding']
            
            all_causes.extend(causes)
            all_effects.extend(effects)
            all_cause_embeddings.append(cause_emb.cpu().numpy())
            all_effect_embeddings.append(effect_emb.cpu().numpy())

        cause_embeddings = np.vstack(all_cause_embeddings)
        effect_embeddings = np.vstack(all_effect_embeddings)

        index_cause = faiss.IndexFlatIP(cause_embeddings.shape[1])
        index_cause.add(cause_embeddings)
        
        index_effect = faiss.IndexFlatIP(effect_embeddings.shape[1])
        index_effect.add(effect_embeddings)
        
        correct_cause = 0
        correct_effect = 0
        
        for i in tqdm(range(len(all_causes)), desc="Evaluating"):
            D, I = index_effect.search(cause_embeddings[i].reshape(1, -1), 1)
            if all_effects[I[0][0]] == all_effects[i]:
                correct_cause += 1
        
        for i in tqdm(range(len(all_effects)), desc="Evaluating"):
            D, I = index_cause.search(effect_embeddings[i].reshape(1, -1), 1)
            if all_causes[I[0][0]] == all_causes[i]:
                correct_effect += 1
    
        
        cause_accuracy = correct_cause / len(all_causes)
        effect_accuracy = correct_effect / len(all_effects)
        
        accuracy = (cause_accuracy + effect_accuracy) / 2
        
        return accuracy


if __name__ == "__main__":
    main()
