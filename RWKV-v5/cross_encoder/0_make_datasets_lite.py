import datasets
from datasets import load_dataset
import os
current_dir = os.path.dirname(__file__)
#set huggingface cache dir to current directory's cache_data
os.environ["HF_HOME"] = os.path.join(current_dir, "cache_data")
script_file = os.path.join(current_dir, "cross_encoder_ds.py")
dataset = load_dataset(script_file,"chinese", data_dir="/home/gpu/data/sdb1/downloads/google_data",num_proc=80,cache_dir=os.path.join(current_dir, "cache_data"))
print(dataset['train'][1])
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
# Initialize the tokenizer outside the function
dict_file = os.path.join(current_dir, "rwkv_vocab_v20230424.txt")
tokenizer = TRIE_TOKENIZER(dict_file)
cls_id = 1
sep_id = 2
max_length = 512
pad_id = 0
def convert_example(examples):
    queries = examples['query']
    positives = examples['positive']
    negatives = examples['negative']
    positive_ids_list = []
    negative_ids_list = []
    for i in range(len(queries)):
        positive_ids = tokenizer.encode(queries[i]) + [sep_id] + tokenizer.encode(positives[i])+[cls_id]
        negative_ids = tokenizer.encode(queries[i]) + [sep_id] + tokenizer.encode(negatives[i])+[cls_id]
        #pad to max_length
        positive_len = len(positive_ids)
        negative_len = len(negative_ids)
        if positive_len <= max_length:
            positive_ids += [pad_id] * (max_length - positive_len)
        else:
            continue
        if negative_len <= max_length:
            negative_ids += [pad_id] * (max_length - negative_len)
        else:
            continue
        positive_ids_list.append(positive_ids)
        negative_ids_list.append(negative_ids)
    return {"positive": positive_ids_list, "negative": negative_ids_list}

dataset = dataset.map(convert_example, batched=True, num_proc=80, remove_columns=['query', 'positive', 'negative'])

save_dir = '/home/gpu/data/sdb1/mmarco_chinese/rwkv_tokenized_ids'

dataset.save_to_disk(save_dir)

print(dataset['train'][1])
