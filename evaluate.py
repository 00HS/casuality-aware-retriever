#%%
import os
import json
import numpy as np
from tqdm.auto import tqdm

import faiss
import torch
from transformers import BertTokenizer

from utils_cawai import set_seed, load_checkpoint, load_config, parse_args, load_jsonl


def encode_texts_in_batches(texts, tokenizer, model, device, batch_size, model_run_name):
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i + batch_size]
            if model_run_name == 'cawai_dpr':
                encoded_batch = tokenizer(batch, truncation=True, padding=True, max_length=256, return_tensors='pt').to(device)
                outputs = model(encoded_batch)
            else:
                features = model.tokenize(batch)
                features = {k: v.to(device) for k, v in features.items()}
                outputs = model(features)['sentence_embedding']
            embeddings.append(outputs.cpu().numpy())
    return np.vstack(embeddings)


def save_partial_embeddings(texts, model, tokenizer, device, batch_size, output_file, model_run_name):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        with torch.no_grad():
            if model_run_name == 'cawai_dpr':
                encoded_batch = tokenizer(batch, truncation=True, padding=True, max_length=256, return_tensors='pt').to(device)
                outputs = model(encoded_batch)
            else:
                features = model.tokenize(batch)
                features = {k: v.to(device) for k, v in features.items()}
                outputs = model(features)['sentence_embedding']
        embeddings.append(outputs.cpu().numpy())
    np.save(output_file, np.vstack(embeddings))


def save_chunked_embeddings_and_create_index(corpus, model, tokenizer, device, batch_size, chunk_size, output_prefix, model_run_name):
    embedding_files = []

    for idx, chunk_start in enumerate(tqdm(range(0, len(corpus), chunk_size), desc="Processing chunks")):
        chunk = corpus[chunk_start:chunk_start + chunk_size]
        output_file = f"embedding/cawai/{output_prefix}_part{idx}.npy"
        save_partial_embeddings(chunk, model, tokenizer, device, batch_size, output_file, model_run_name)
        embedding_files.append(output_file)

    index = None
    for file in embedding_files:
        partial_embeddings = np.load(file)
        if index is None:
            index = faiss.IndexFlatIP(partial_embeddings.shape[1])
        index.add(partial_embeddings)
    faiss.write_index(index, f"embedding/cawai/{output_prefix}.index")
    return index


def evaluate_retrieval(query_texts, query_embeddings, corpus_texts, index, passage_texts, query_type, answer_type, k):
    D, I = index.search(query_embeddings, k)

    correct_counts = [0] * k
    results = []

    for i in tqdm(range(len(query_embeddings))):
        top_k_indices = I[i]
        correct_answer = passage_texts[i]

        is_correct = [False] * k
        predicted_answers = []

        for j, idx in enumerate(top_k_indices):
            predicted_answer = corpus_texts[idx]
            predicted_answers.append(predicted_answer)

            if predicted_answer == correct_answer:
                is_correct[j] = True
                for m in range(j, k):
                    correct_counts[m] += 1

        result = {
            f"{query_type}": query_texts[i],
            f"correct_{answer_type}": correct_answer,
            f"predicted_{answer_type}s": predicted_answers,
            "is_correct": is_correct
        }
        results.append(result)

    accuracies = [count / len(query_embeddings) for count in correct_counts]
    return accuracies, results


def main():
    args = parse_args()
    config = load_config(args.config_path)

    set_seed(args.random_seed)
    os.makedirs('embedding/cawai', exist_ok=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model_run_name = args.model_run_name

    test_data = load_jsonl('dataset/e_care/test.jsonl')
    causes = [item['cause'] for item in test_data]
    effects = [item['effect'] for item in test_data]

    cause_corpus, effect_corpus = causes, effects

    if model_run_name == 'cawai_dpr':
        from model.cawai_dpr.model import Encoder, CausalEncoder
        tokenizer = BertTokenizer.from_pretrained(args.model_name)
    else:
        from model.cawai_gtr.model import Encoder, CausalEncoder
        tokenizer = None

    cause_encoder = Encoder(args.model_name, device)
    effect_encoder = Encoder(args.model_name, device)
    semantic_encoder = Encoder(args.model_name, device)
    model = CausalEncoder(cause_encoder, effect_encoder, semantic_encoder, device).to(device)

    checkpoint = f'checkpoint/{model_run_name}.pth'
    load_checkpoint(model, optimizer=None, checkpoint_path=checkpoint)
    model.eval()

    # Save corpus embeddings
    print("Encoding cause_corpora...")
    cause_corpus_index = save_chunked_embeddings_and_create_index(cause_corpus, model.cause_encoder, tokenizer, device, 64, 500000, f'{model_run_name}_cause_corpus', model_run_name)

    print("Encoding effect_corpora...")
    effect_corpus_index = save_chunked_embeddings_and_create_index(effect_corpus, model.effect_encoder, tokenizer, device, 64, 500000, f'{model_run_name}_effect_corpus', model_run_name)

    print("Encoding queries...")
    cause_query_embeddings = encode_texts_in_batches(causes, tokenizer, model.cause_encoder, device, 256, model_run_name)
    effect_query_embeddings = encode_texts_in_batches(effects, tokenizer, model.effect_encoder, device, 256, model_run_name)

    corpus_pairs = [
        (causes, cause_query_embeddings, effect_corpus, effect_corpus_index, effects, "results/e-care", "cause", "effect"),
        (effects, effect_query_embeddings, cause_corpus, cause_corpus_index, causes, "results/e-care", "effect", "cause"),
    ]

    for query_texts, query_embeddings, corpus_texts, corpus_index, correct_texts, folder_name, query_type, answer_type in corpus_pairs:
        print(f"Evaluating {folder_name} {query_type}...")
        accuracies, results = evaluate_retrieval(query_texts, query_embeddings, corpus_texts, corpus_index, correct_texts, query_type, answer_type, k=10)

        print(f"{folder_name} {query_type} Retrieval Accuracies:")
        for i, acc in enumerate(accuracies, 1):
            print(f"  Top-{i} Accuracy: {acc:.3f}")

        output_dir = os.path.join(folder_name, f"query_{query_type}")
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, f"{model_run_name}_top10.json")
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Results saved to {output_file_path}")

    print("Evaluation complete.")


if __name__ == "__main__":
    main()
