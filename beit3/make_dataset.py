from datasets import HatefulMemesDataset
from transformers import XLMRobertaTokenizer
import json

#---------------------------------
# Begin original code contribution
#---------------------------------

def combine_seen_unseen(split, aug=False):
    entity = ".entity" if aug else ""
    seen_path = f"hm_data/{split}_seen{entity}.jsonl"
    unseen_path = f"hm_data/{split}_unseen{entity}.jsonl"

    with open(seen_path, 'r') as file:
        seen_data = {line.strip() for line in file}

    with open(unseen_path, 'r') as file:
        unseen_data = {line.strip() for line in file}
    
    combined_data = seen_data.union(unseen_data)

    with open(f'hm_data/{split}_seen_unseen{entity}.jsonl', 'w') as file:
        for item in combined_data:
            file.write(f"{item}\n")

def add_aug_test_labels(split):
    with open(f'hm_data/{split}_seen_unseen.jsonl', 'r') as f:
        unaug_data = [json.loads(l) for l in f]
    with open(f'hm_data/{split}_seen_unseen.entity.jsonl', 'r') as f:
        aug_data = [json.loads(l) for l in f]

    labels = {data['img']: data['label'] for data in unaug_data}

    for data in aug_data:
        if data['img'] in labels:
            data['label'] = labels[data['img']]

    with open(f'hm_data/{split}_seen_unseen.entity.jsonl', 'w') as f:
        for data in aug_data:
            f.write(json.dumps(data) + '\n')

combine_seen_unseen("dev", aug=False)
combine_seen_unseen("dev", aug=True)
combine_seen_unseen("test", aug=False)
combine_seen_unseen("test", aug=True)
add_aug_test_labels("dev")
add_aug_test_labels("test")

tokenizer = XLMRobertaTokenizer("beit3.spm")

HatefulMemesDataset.make_dataset_index(
    data_path="hm_data", 
    tokenizer=tokenizer
)

HatefulMemesDataset.make_dataset_index(
    data_path="hm_data", 
    tokenizer=tokenizer,
    aug=True
)

#-------------------------------
# End original code contribution
#-------------------------------