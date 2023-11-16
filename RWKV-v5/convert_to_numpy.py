import numpy as np
import json
import os

from tqdm import tqdm

def parse_json(filename,outputfile):
    with open(filename, 'r') as f:
        data = json.load(f)
    
    output_data = []
    from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
    tokenizer = TRIE_TOKENIZER(os.path.dirname(os.path.abspath(__file__)) + '/rwkv_vocab_v20230424.txt' )
    lengths = []
    for item in tqdm(data, desc="Processing data"):
        ids = tokenizer.encode(item)
        output_data.append(ids)
        lengths.append(len(ids))
    max_len = max(lengths)
    min_len = min(lengths)
    avg_len = sum(lengths) / len(lengths)
    mean_len = np.mean(lengths)
    np_output_data = np.array(output_data)
    np.save(outputfile, np_output_data)
    print(f"Max length of ids: {max_len}")
    print(f"Min length of ids: {min_len}")
    print(f"Average length of ids: {avg_len}")
    print(f"Mean length of ids: {mean_len}")
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default='data.json', help='json file')
    parser.add_argument('--output', type=str, default='output.txt', help='output file')
    args = parser.parse_args() 
    parse_json(args.filename,args.output)