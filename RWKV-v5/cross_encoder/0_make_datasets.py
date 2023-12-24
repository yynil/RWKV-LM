import datasets
from datasets import load_dataset
import os
current_dir = os.path.dirname(__file__)
script_file = os.path.join(current_dir, "cross_encoder_ds.py")
dataset = load_dataset(script_file,"chinese", data_dir="/home/yueyulin/下载/google")
print(dataset['train'][1])

def convert_example(examples):
    queries = examples['query']
    positives = examples['positive']
    negatives = examples['negative']
    positive_texts = [f"【问题：{queries[i]}\n文档：{positives[i]}\n】" for i in range(len(queries))]
    negative_texts = [f"【问题：{queries[i]}\n文档：{negatives[i]}\n】" for i in range(len(queries))]
    return {"positive": positive_texts, "negative": negative_texts}

dataset = dataset.map(convert_example, batched=True, num_proc=24, remove_columns=['query', 'positive', 'negative'])

save_dir = '/media/yueyulin/bigdata/ds/mmarco_chinese/texts'

dataset.save_to_disk(save_dir)

print(dataset['train'][1])

from functools import partial
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER


# Initialize the tokenizer outside the function
dict_file = os.path.join(current_dir, "rwkv_vocab_v20230424.txt")
tokenizer = TRIE_TOKENIZER(dict_file)
cls_id = 1
def tokenize_function(examples):
    positives = examples['positive']
    negatives = examples['negative']
    positive_tokens = [tokenizer.encode(text)+[cls_id] for text in positives]
    negative_tokens = [tokenizer.encode(text)+[cls_id] for text in negatives]
    return {"positive": positive_tokens, "negative": negative_tokens}


dataset = dataset.map(tokenize_function, batched=True, num_proc=24, remove_columns=['positive', 'negative'])
save_dir = '/media/yueyulin/bigdata/ds/mmarco_chinese/rwkv_tokenized_ids'
dataset.save_to_disk(save_dir)
print(dataset['train'][1])
print(len(dataset['train']))
max_length = 128
pad_id = 0

def truncate_and_pad_examples(examples):
    positives = examples['positive']
    negatives = examples['negative']
    filtered_positives = []
    filtered_negatives = []
    batch_size = len(positives)
    for i in range(batch_size):
        positive = positives[i]
        negative = negatives[i]
        postive_len = len(positive)
        negative_len = len(negative)
        if postive_len <= max_length and negative_len <= max_length:
            positive = positive + [pad_id] * (max_length - postive_len)
            negative = negative + [pad_id] * (max_length - negative_len)
            filtered_positives.append(positive)
            filtered_negatives.append(negative)
    return {"positive": filtered_positives, "negative": filtered_negatives}

dataset = dataset.map(truncate_and_pad_examples, batched=True, num_proc=24, remove_columns=['positive', 'negative'])
print(dataset['train'][1])
print(len(dataset['train']))


