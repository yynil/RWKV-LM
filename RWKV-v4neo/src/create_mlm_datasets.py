from datasets import load_dataset
import random
from typing import List, Tuple
import jieba
import multiprocessing as mp

# Create a datasets with fixed tokenized length, padding with pad_id
# Transform the input dataset's text column into a src and dst input_ids, the src is a MLM input_ids with 80% unmasked, 10% masked, 10% random ids.

tokenizer = None

def create_tokenizer():
    import rwkv
    from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
    import os,inspect
    module = inspect.getmodule(TRIE_TOKENIZER)
    print(module.__file__)
    tokenizer_file = os.path.dirname(os.path.abspath(module.__file__)) + '/rwkv_vocab_v20230424.txt'
    tokenizer = TRIE_TOKENIZER(tokenizer_file)
    return tokenizer

def get_tokenizer():
    global tokenizer
    if tokenizer is None:
        print(f"create tokenizer for {mp.current_process().pid}")
        tokenizer = create_tokenizer()
    return tokenizer


def convert_input_ids_2_words_ids_mp(words :List[str]):
    """
    words is the list of words in the sentence
    """
    tokenizer = get_tokenizer()
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
        if r < mlm_probability:
            #keep the word orginal
            mlm_input_ids.extend(word_ids)
            mlm_labels.extend([-100] * len(word_ids))
        elif r < mlm_probability + random_probabilty:
            #replace the word with mask_id
            mlm_input_ids.extend([mask_id] * len(word_ids))
            mlm_labels.extend(word_ids)
        else:
            #replace the word with random id between min_id and max_id inclusively
            random_ids = [random.randint(min_id, max_id) for _ in range(len(word_ids))]
            mlm_input_ids.extend(random_ids)
            mlm_labels.extend(word_ids)
    return mlm_input_ids, mlm_labels

def convert_llm_to_mlm_dataset(llm_dataset, tokenizer, fixed_length, pad_id,mask_token_id,min_id = 10250, max_id=18493):
    # fixed_length: the fixed length of the input_ids
    # pad_id: the id of the padding token
    # tokenizer: the tokenizer used to tokenize the text
    # llm_dataset: the dataset with the text column to be converted
    # return: the dataset with the text column converted to src and dst input_ids
    print(jieba.lcut('中华人民共和国的人民在种花。'))
    
    def convert_to_mlm(example):
        # print(f"convert_to_mlm {mp.current_process().pid}")
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
            mlm_input_ids, mlm_labels = generate_mlm_data(convert_input_ids_2_words_ids_mp(words), mask_token_id, min_id, max_id)
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
    


    return llm_dataset.map(convert_to_mlm, remove_columns=llm_dataset.features, batched=True,num_proc=24)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    ds_name = 'liwu/MNBVC'
    parser.add_argument('--subsets', type=str, default=['law_judgement','gov_xuexiqiangguo','gov_report','co_ann_report','news_peoples_daily','wikipedia'],nargs='+', help='Subsets to use')
    parser.add_argument('--fixed_length', type=int, default=512, help='Fixed length of the input_ids')
    parser.add_argument('--pad_id', type=int, default=1, help='Id of the padding token')
    parser.add_argument('--ignore_index', type=int, default=-100, help='Id of the padding token')
    parser.add_argument('--mask_token_id', type=int, default=3, help='Id of the mask token')
    parser.add_argument('--min_id', type=int, default=10250, help='Min id of the random token')
    parser.add_argument('--max_id', type=int, default=18493, help='Max id of the random token')
    parser.add_argument('--output_dir',type=str, default='tmp', help='Output directory')
    parser.add_argument('--cache_dir',type=str, default='/mnt/d/hf_cache', help='Cache directory')
    parser.add_argument('--do_export',default=False, action='store_true', help='Export the datasets')
    parser.add_argument('--local_cache',default=False, action='store_true', help='Use local cache')
    parser.add_argument('--local_dir',type=str,default='', help='Local directory')
    args = parser.parse_args()
    import os
    for subset in args.subsets:
        split = 'train'
        if args.local_cache:
            print(f'load from local cache {args.local_dir}')
            train_ds = load_dataset(args.local_dir,split=split,streaming=False)
        else:
            print(f'load dataset {ds_name} {subset} {split}')
            train_ds = load_dataset(ds_name, subset, split=split, streaming=False,cache_dir=args.cache_dir)
        features = train_ds.features
        print(features)
        if 'text' not in features:
            if '段落' in features:
                #map the data to text
                def combine_paras(example):
                    paras = example['段落']
                    text = ''
                    for para in paras:
                        text += para['内容']
                    return {'text': text}
                train_ds = train_ds.map(combine_paras, batched=False,remove_columns=features)
            else:
                raise Exception('No text column in the dataset')
        else:
            #remove other columns other than text
            removed_columns = [feature for feature in features if feature != 'text']
            train_ds = train_ds.map(lambda example: {'text': example['text']}, batched=True,remove_columns=removed_columns, num_proc=24)
        print(train_ds)
        converted_ds = convert_llm_to_mlm_dataset(train_ds, tokenizer, args.fixed_length, args.pad_id, args.mask_token_id, args.min_id, args.max_id)
        output_path = os.path.join(args.output_dir, f'{ds_name}_{subset}_{split}_mlm_{args.fixed_length}_pad_{args.pad_id}_mask_{args.mask_token_id}_min_{args.min_id}_max_{args.max_id}')
        print(converted_ds[0])
        if args.do_export:
            converted_ds.save_to_disk(output_path)
            print(f'Saved {len(converted_ds)} examples to {output_path}')
        else:
            print(f'Not saving to disk')

from datasets import load_dataset
import random
from typing import List, Tuple
import jieba
import multiprocessing as mp

# Create a datasets with fixed tokenized length, padding with pad_id
# Transform the input dataset's text column into a src and dst input_ids, the src is a MLM input_ids with 80% unmasked, 10% masked, 10% random ids.

tokenizer = None

def create_tokenizer():
    import rwkv
    from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
    import os,inspect
    module = inspect.getmodule(TRIE_TOKENIZER)
    print(module.__file__)
    tokenizer_file = os.path.dirname(os.path.abspath(module.__file__)) + '/rwkv_vocab_v20230424.txt'
    tokenizer = TRIE_TOKENIZER(tokenizer_file)
    return tokenizer

def get_tokenizer():
    global tokenizer
    if tokenizer is None:
        print(f"create tokenizer for {mp.current_process().pid}")
        tokenizer = create_tokenizer()
    return tokenizer


def convert_input_ids_2_words_ids_mp(words :List[str]):
    """
    words is the list of words in the sentence
    """
    tokenizer = get_tokenizer()
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
    print(jieba.lcut('中华人民共和国的人民在种花。'))
    
    def convert_to_mlm(example):
        # print(f"convert_to_mlm {mp.current_process().pid}")
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
            mlm_input_ids, mlm_labels = generate_mlm_data(convert_input_ids_2_words_ids_mp(words), mask_token_id, min_id, max_id)
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
    


    return llm_dataset.map(convert_to_mlm, remove_columns=llm_dataset.features, batched=True,num_proc=24)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    ds_name = 'liwu/MNBVC'
    parser.add_argument('--subsets', type=str, default=['law_judgement','gov_xuexiqiangguo','gov_report','co_ann_report','news_peoples_daily','wikipedia'],nargs='+', help='Subsets to use')
    parser.add_argument('--fixed_length', type=int, default=512, help='Fixed length of the input_ids')
    parser.add_argument('--pad_id', type=int, default=1, help='Id of the padding token')
    parser.add_argument('--ignore_index', type=int, default=-100, help='Id of the padding token')
    parser.add_argument('--mask_token_id', type=int, default=3, help='Id of the mask token')
    parser.add_argument('--min_id', type=int, default=10250, help='Min id of the random token')
    parser.add_argument('--max_id', type=int, default=18493, help='Max id of the random token')
    parser.add_argument('--output_dir',type=str, default='tmp', help='Output directory')
    parser.add_argument('--cache_dir',type=str, default='/mnt/d/hf_cache', help='Cache directory')
    parser.add_argument('--do_export',default=False, action='store_true', help='Export the datasets')
    parser.add_argument('--local_cache',default=False, action='store_true', help='Use local cache')
    parser.add_argument('--local_dir',type=str,default='', help='Local directory')
    args = parser.parse_args()
    import os
    for subset in args.subsets:
        split = 'train'
        if args.local_cache:
            print(f'load from local cache {args.local_dir}')
            train_ds = load_dataset(args.local_dir,split=split,streaming=False)
        else:
            print(f'load dataset {ds_name} {subset} {split}')
            train_ds = load_dataset(ds_name, subset, split=split, streaming=False,cache_dir=args.cache_dir)
        features = train_ds.features
        print(features)
        if 'text' not in features:
            if '段落' in features:
                #map the data to text
                def combine_paras(example):
                    paras = example['段落']
                    text = ''
                    for para in paras:
                        text += para['内容']
                    return {'text': text}
                train_ds = train_ds.map(combine_paras, batched=False,remove_columns=features)
            else:
                raise Exception('No text column in the dataset')
        else:
            #remove other columns other than text
            removed_columns = [feature for feature in features if feature != 'text']
            train_ds = train_ds.map(lambda example: {'text': example['text']}, batched=True,remove_columns=removed_columns, num_proc=24)
        print(train_ds)
        converted_ds = convert_llm_to_mlm_dataset(train_ds, tokenizer, args.fixed_length, args.pad_id, args.mask_token_id, args.min_id, args.max_id)
        output_path = os.path.join(args.output_dir, f'{ds_name}_{subset}_{split}_mlm_{args.fixed_length}_pad_{args.pad_id}_mask_{args.mask_token_id}_min_{args.min_id}_max_{args.max_id}')
        print(converted_ds[0])
        if args.do_export:
            converted_ds.save_to_disk(output_path)
            print(f'Saved {len(converted_ds)} examples to {output_path}')
        else:
            print(f'Not saving to disk')

