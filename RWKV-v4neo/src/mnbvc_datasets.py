
from datasets import Dataset
from datasets import load_dataset
from numpy import pad
import torch
import random
from typing import List, Tuple
import jieba
#Create a datasets with fixed tokenized length, padding with pad_id
#Transform the input dataset's text column into a src and dst input_ids, the src is a MLM input_ids with 80% unmasked, 10% masked, 10% random ids.

def convert_input_ids_2_words_ids(words :List[str], tokenizer):
    """
    words is the list of words in the sentence
    """
    words_ids = []
    for word in words:
        word_ids = tokenizer.encode(word)
        words_ids.append(word_ids)
    return words_ids

def generate_mlm_data(words_ids :List[List[int]], mask_id :int = 3,min_id = 10250, max_id=18493, mlm_probability :float = 0.8, random_probabilty :float = 0.1) -> Tuple[List[int], List[int]]:
    """
    words_ids is the list of word ids in the sentence

    """
    mlm_input_ids = []
    mlm_labels = []
    for word_ids in words_ids:
        r = random.random()
        mlm_labels.extend(word_ids)
        if r < mlm_probability:
            #keep the word orginal
            mlm_input_ids.extend(word_ids)
        elif r < mlm_probability + random_probabilty:
            #replace the word with mask_id
            mlm_input_ids.extend([mask_id] * len(word_ids))
        else:
            #replace the word with random id between min_id and max_id inclusively
            random_ids = [random.randint(min_id, max_id) for _ in range(len(word_ids))]
            mlm_input_ids.extend(random_ids)
    return mlm_input_ids, mlm_labels

def convert_llm_to_mlm_dataset(llm_dataset, tokenizer, fixed_length, pad_id,mask_token_id,min_id = 10250, max_id=18493):
    # fixed_length: the fixed length of the input_ids
    # pad_id: the id of the padding token
    # tokenizer: the tokenizer used to tokenize the text
    # llm_dataset: the dataset with the text column to be converted
    # return: the dataset with the text column converted to src and dst input_ids
    def convert_to_mlm(example):
        # example: the example in the dataset
        # return: the example with the text column converted to src and dst input_ids
        ignore_index = -100
        pad_direction = 'right'
        texts = example['text']
        final_src_input_ids = []
        final_dst_input_ids = []
        for text in texts:
            words = jieba.lcut(text)
            # print(words)
            mlm_input_ids, mlm_labels = generate_mlm_data(convert_input_ids_2_words_ids(words, tokenizer), mask_token_id, min_id, max_id)
            #split the mlm_input_ids and mlm_labels into fixed_length, the last part will be padded with pad_id
            src_input_ids = []
            dst_input_ids = []
            offset = 0
            while offset + fixed_length < len(mlm_input_ids):
                src_input_ids.append(mlm_input_ids[offset:offset+fixed_length])
                dst_input_ids.append(mlm_labels[offset:offset+fixed_length])
                offset += fixed_length
            
            if offset < len(mlm_input_ids):
                padded_length = fixed_length - (len(mlm_input_ids) - offset)
                if pad_direction == 'right':
                    src_input_ids.append(mlm_input_ids[offset:] + [pad_id] * padded_length)
                    dst_input_ids.append(mlm_labels[offset:] + [ignore_index] * padded_length)
                else:
                    src_input_ids.append([pad_id] * padded_length + mlm_input_ids[offset:])
                    dst_input_ids.append([ignore_index] * padded_length + mlm_labels[offset:])
            final_src_input_ids.extend(src_input_ids)
            final_dst_input_ids.extend(dst_input_ids)
        return {'src_input_ids': final_src_input_ids, 'dst_input_ids': final_dst_input_ids}

        # mlm_input_ids = mlm_input_ids[:fixed_length]
        # mlm_labels = mlm_labels[:fixed_length]
        # padded_length = fixed_length - len(mlm_input_ids)
        # if pad_direction == 'right':
        #     mlm_input_ids = mlm_input_ids + [pad_id] * padded_length
        #     mlm_labels = mlm_labels + [ignore_index] * padded_length
        # else:
        #     mlm_input_ids = [pad_id] * padded_length + mlm_input_ids
        #     mlm_labels = [ignore_index] * padded_length + mlm_labels
        # return {'src_input_ids': src_input_ids, 'dst_input_ids': dst_input_ids}

    return llm_dataset.map(convert_to_mlm, remove_columns=llm_dataset.features, batched=True)
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.features = dataset.features
        self.num_rows = len(dataset)
    
    def __len__(self):
        return self.num_rows
    
    def __getitem__(self, index):
        data = self.dataset[index]
        return torch.tensor(data['src_input_ids'],dtype=torch.long), torch.tensor(data['dst_input_ids'],dtype=torch.long)


if __name__ == '__main__':

    #create a mock up datasets with text column 
    mockup_data = {'text': ['中华人民共和国于今天成立了。', '明亡仕清。历官河间府知府。顺治二年（1645年），升任山东按察使司副使、天津饷道，十一月调江西按察使司副使、提调学政，四年七月升福建布政使司参政兼按察使司佥事、分巡漳南道。']}
    mockup_dataset = Dataset.from_dict(mockup_data)
    print(mockup_dataset)
    import rwkv
    from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
    import os,inspect
    module = inspect.getmodule(TRIE_TOKENIZER)
    print(module.__file__)
    tokenizer_file = os.path.dirname(os.path.abspath(module.__file__)) + '/rwkv_vocab_v20230424.txt'
    tokenizer = TRIE_TOKENIZER(tokenizer_file)
    fixed_length = 128
    pad_id = 1
    mask_token_id = 3
    mlm_dataset = convert_llm_to_mlm_dataset(mockup_dataset, tokenizer, fixed_length, pad_id,mask_token_id)
    print(mlm_dataset)
    print(mlm_dataset[0])

    # ds_name = 'liwu/MNBVC'
    # data_name = 'gov_report'
    # dataset = load_dataset(ds_name, data_name, split='train', streaming=True)

    # print(next(iter(dataset)))
    wikipedia_dir = "/home/setu/.cache/huggingface/datasets/liwu___mnbvc/wikipedia/0.0.1/41825a8f3da6eceada2c3215d3eafe5e07bfee42adc8ac828463e14328f2cf91/"
    dataset = load_dataset(wikipedia_dir, split='train', streaming=False)
    print(dataset)
    features = dataset.features
    print(dataset[0])
    # dataset = dataset.select(range(100))
    def combine_paras(example):
        paras = example['段落']
        text = ''
        for para in paras:
            text += para['内容']
        return {'text': text}
    dataset = dataset.map(combine_paras, batched=False,remove_columns=features)
    print(dataset)
    print(dataset[0])
    mlm_dataset = convert_llm_to_mlm_dataset(dataset, tokenizer, fixed_length, pad_id,mask_token_id)
    print(mlm_dataset[0])
    store_dir = f'/home/setu/tmp/mnbvc/wikipedia_{fixed_length}'
    mlm_dataset.save_to_disk(store_dir)
    print('done')