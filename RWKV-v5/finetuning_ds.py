from datasets import Dataset,load_dataset
import os
def convert_json_file_to_ds(
    tokenizer,
    input_json_file
):
    ds = load_dataset('json', data_files=input_json_file)
    ds = ds.map(lambda example: {'input_str': f"User: 根据以下内容，撰写文章标题。\n{example['content']}\nAssistant: 类型：{example['dataType']}\n标题：{example['title']}"},remove_columns=ds['train'].features)
    
    ds = ds.map(lambda example: {'input_ids': tokenizer.encode(example['input_str'])})
    
    return ds

if __name__ == '__main__':
    filename = '/Users/yueyulin/Downloads/part-202101281a.json'
    from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
    tokenizer = TRIE_TOKENIZER(os.path.dirname(os.path.abspath(__file__)) + '/rwkv_vocab_v20230424.txt' )
    ds = convert_json_file_to_ds(tokenizer,filename)
    print(ds)
    print(ds['train'][0])