import json
from functools import partial
from torch.utils.data import Dataset, DataLoader

class CausalQA(Dataset):
    def __init__(self, data):
        self.data = data
        self.questions = [item['question'] for item in data]  
        self.passages = [item['passage_processed'] for item in data] 

    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        return {
            "question": self.questions[idx],
            "passage": self.passages[idx]
        }

def collate_fn(batch_size, batch):
    batch_data = {
        "question": [],
        "passage": []
    }

    for item in batch:
        batch_data["question"].append(item["question"])
        batch_data["passage"].append(item["passage"])

    if len(batch_data["question"]) == batch_size:
        return batch_data
    else:
        return None

def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def parse_data(config, dtype):
    if dtype == "train":
        data_files = config['dataset']['train_files']
    elif dtype == "dev":
        data_files = config['dataset']['dev_file']
    else:  # dtype == "test"
        data_files = [config['dataset']['test_file']]
    
    data = []
    for file_path in data_files:
        data.extend(load_jsonl(file_path))

    data_loader = DataLoader(
        CausalQA(data), 
        batch_size=config['dataset']['batch_size'], 
        shuffle=True if dtype == "train" else False, 
        num_workers=5,
        pin_memory=True,
        drop_last=(dtype == "train"),
        persistent_workers=True,
        collate_fn=partial(collate_fn, config['dataset']['batch_size'])
    )
    
    data_loader = list(filter(lambda x: x is not None, data_loader))        

    return data_loader
