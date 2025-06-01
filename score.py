import os
import json

RESULTS_DIR = 'results/e-care'
FILE_SUFFIX = '_top10.json'

def calculate_metric(json_data, metric_type):
    count = 0
    total_reciprocal_rank = 0.0
    num_examples = len(json_data)

    for example in json_data:
        is_correct = example['is_correct']

        if metric_type == 'hit1':
            if len(is_correct) > 0 and is_correct[0]:
                count += 1
        elif metric_type == 'hit10':
            if any(is_correct[:10]):
                count += 1
        elif metric_type == 'mrr':
            for idx, correct in enumerate(is_correct[:10]):
                if correct:
                    rank = idx + 1
                    total_reciprocal_rank += 1.0 / rank
                    break

    if metric_type == 'mrr':
        return round(total_reciprocal_rank / num_examples if num_examples > 0 else 0.0, 3)
    else:
        return round(count / num_examples if num_examples > 0 else 0.0, 3)

def print_scores(folder_name, query_type, model_name, hit1, hit10, mrr):
    print(f"{folder_name}+{query_type}+{model_name}: & {hit1 * 100:.1f} & {hit10 * 100:.1f} & {mrr * 100:.1f}")

def calculate_scores_for_directory(base_dir):
    for root, dirs, files in os.walk(base_dir):
        folder_name = os.path.basename(os.path.dirname(root))
        query_type = os.path.basename(root)

        for file in files:
            if file.endswith(FILE_SUFFIX):
                model_name = file.replace(FILE_SUFFIX, '')
                json_path = os.path.join(root, file)
                with open(json_path, 'r') as f:
                    json_data = json.load(f)

                    hit1 = calculate_metric(json_data, 'hit1')
                    hit10 = calculate_metric(json_data, 'hit10')
                    mrr = calculate_metric(json_data, 'mrr')

                    print_scores(folder_name, query_type, model_name, hit1, hit10, mrr)

if __name__ == "__main__":
    calculate_scores_for_directory(RESULTS_DIR)