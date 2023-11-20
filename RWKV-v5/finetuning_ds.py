from datasets import Dataset,load_dataset
import os
import datasets

import torch
from torch.utils.data import Dataset
class MyDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        example = self.ds[index]
        input_ids = example['input_ids']
        target = example['targets']
        return torch.tensor(input_ids, dtype=torch.int), torch.tensor(target, dtype=torch.int)
def convert_json_file_to_ds(
    tokenizer,
    input_json_file, ctx_len=4096
):
    ds = load_json_as_ds(tokenizer, input_json_file, ctx_len)
        
    ds = MyDataset(ds['train'])
    return ds

def load_json_as_ds(tokenizer, input_json_file, ctx_len):
    ctx_len += 1
    if tokenizer is None:
        from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
        tokenizer = TRIE_TOKENIZER(os.path.dirname(os.path.abspath(__file__)) + '/rwkv_vocab_v20230424.txt' )
    def process_example(example):
        input_str = f"User: 根据以下内容，撰写文章标题。\n{example['content']}\nAssistant: 类型：{example['dataType']}\n标题：{example['title']}"
        input_ids = tokenizer.encode(input_str)
        length_of_input_ids = len(input_ids)
        if length_of_input_ids <= ctx_len:
            targets = input_ids[1:length_of_input_ids]+[-100] * (ctx_len - length_of_input_ids)
            input_ids = input_ids[0:length_of_input_ids-1] + [0] * (ctx_len - length_of_input_ids)

        else: 
            input_ids = input_ids[:ctx_len-1]
            targets = input_ids[1:ctx_len]
        return {'input_str': input_str, 'input_ids': input_ids,'targets':targets}

    ds = load_dataset('json', data_files=input_json_file)
    ds = ds.map(process_example, remove_columns=ds['train'].features)

    return ds

def load_ds(ds_dir):
    return datasets.load_from_disk(ds_dir)

def load_datasets(ds_dir):
    return MyDataset(load_ds(ds_dir)['train'])

if __name__ == '__main__':
    filename = '/Users/yueyulin/Downloads/part-202101281a.json'
    from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
    tokenizer = TRIE_TOKENIZER(os.path.dirname(os.path.abspath(__file__)) + '/rwkv_vocab_v20230424.txt' )
    ds = load_json_as_ds(tokenizer,filename,4096)
    output_dir = '/Users/yueyulin/Downloads/ds_tmp'
    os.makedirs(output_dir,exist_ok=True)
    ds.save_to_disk(output_dir)
    # output_dir = '/Users/yueyulin/Downloads/ds_tmp'
    # ds = load_datasets(output_dir)
    # print(ds[0])
    # print(ds)