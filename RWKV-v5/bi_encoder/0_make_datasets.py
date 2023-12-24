import datasets
from datasets import load_dataset
import os

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    #data_file as a str list
    parser.add_argument('--data_files', nargs='+', type=str, help='Input data files')
    parser.add_argument('--save_dir', type=str, help='save dir')

    args = parser.parse_args()

    dataset = load_dataset('json', data_files=args.data_files)
    print(dataset['train'][1])

    from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
    current_dir = os.path.dirname(__file__)
    # Initialize the tokenizer outside the function
    dict_file = os.path.join(current_dir, "rwkv_vocab_v20230424.txt")
    tokenizer = TRIE_TOKENIZER(dict_file)
    cls_id = 1

    def tokenize_function(examples):
        queries = examples['query']
        positives = examples['positive']
        negatives = examples['negative']
        queries_tokens = [tokenizer.encode(text)+[cls_id] for text in queries]
        positive_tokens = [tokenizer.encode(text)+[cls_id] for text in positives]
        negative_tokens = [tokenizer.encode(text)+[cls_id] for text in negatives]
        logits_positives = examples['logits_positive']
        logits_negatives = examples['logits_negative']
        return {"query": queries_tokens, "positive": positive_tokens, "negative": negative_tokens, "logits_positive": logits_positives, "logits_negative": logits_negatives}
    
    dataset = dataset.map(tokenize_function, batched=True, num_proc=24, remove_columns=['query', 'positive', 'negative', 'logits_positive', 'logits_negative'])
    print(dataset['train'][1])
    os.makedirs(args.save_dir, exist_ok=True)
    dataset.save_to_disk(args.save_dir)