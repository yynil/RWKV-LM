import datasets
import os
from tqdm import tqdm
import json
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER


def load_dataset(input_file, cache_dir=None,max_length=512):
    if cache_dir is None:
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.join(current_file_dir, 'cache')
    print('Loading jsonl from ', input_file, ' with cache_dir ', cache_dir)
    dataset = datasets.load_dataset('json', data_files=input_file, cache_dir=cache_dir)
    print(dataset)
    dataset = dataset['train']
    print(dataset[0])
    print(f'Loaded dataset with {len(dataset)} samples')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    trie_tokenizer = TRIE_TOKENIZER(os.path.join(current_dir, 'rwkv_vocab_v20230424.txt'))
    def tokenize_and_padding(examples):
        inputs = examples['input']
        targets = examples['target']
        input_ids = []
        target_ids = []
        eos_id = 2
        pad_id = 0
        for i in range(len(inputs)):
            input = inputs[i]
            target = targets[i]
            input_id = trie_tokenizer.encode(input)
            target_id = trie_tokenizer.encode(target)
            whole_input_ids = input_id  + target_id
            whole_target_ids = [-100]*(len(input_id)-1)+target_id+[eos_id]
            if len(whole_input_ids) < max_length:
                whole_input_ids = whole_input_ids + [pad_id]*(max_length-len(whole_input_ids))
                whole_target_ids = whole_target_ids + [-100]*(max_length-len(whole_target_ids))
                input_ids.append(whole_input_ids)
                target_ids.append(whole_target_ids)
        return {'input_ids': input_ids, 'labels': target_ids}
    num_cpus = os.cpu_count()
    print(f'Using {num_cpus} cpus to process the dataset')
    dataset = dataset.map(tokenize_and_padding, batched=True,batch_size=4, num_proc=1,remove_columns=['input', 'target', 'kind'])
    print(f'Finished processing dataset with {len(dataset)} samples')
    print(dataset)
    print(dataset[0])
    return dataset
def extract_inputs_targets_worker(output_files_name):
    file_index = int(output_files_name.split('.')[-1])
    print(f'Processing file {file_index} with name {output_files_name}')
    with open(output_files_name, 'r') as f:
        lines = f.readlines()
    data = []
    for line in tqdm(lines):
        conversation_obj = json.loads(line)
        category = conversation_obj['category']
        conversation_list = conversation_obj['conversation']
        for conversation in conversation_list:
            human = conversation['human']
            assistant = conversation['assistant']
            input = human 
            target = assistant
            data.append({'input': input, 'target': target, 'kind': category})
    return data
def extract_inputs_targets(input_file, output_file):
    #first split the input file into num_cpus files by lines
    num_cpus = os.cpu_count()
    output_files_names = [f'{input_file}.{i}' for i in range(num_cpus)]
    

    output_files = [open(f, 'w') for f in output_files_names]
    with open(input_file, 'r') as f:
        lines = f.readlines()
    from tqdm import tqdm
    index = 0
    for line in tqdm(lines):
        output_files[index % num_cpus].write(line)
        index += 1
    for f in output_files:
        f.close()
    import multiprocessing
    pool = multiprocessing.Pool(num_cpus)
    all_data = pool.map(extract_inputs_targets_worker, output_files_names)
    print('Finished processing, all data len is ', len(all_data))
    print(f'merging all data to {output_file}')
    with open(output_file, 'w') as f:
        index = 0
        for data in all_data:
            for d in tqdm(data, desc=f'Writing data with index {index}'):
                f.write(json.dumps(d,ensure_ascii=False) + '\n')
            index += 1
    print('Finished merging')
    #remove the temporary files
    for f in output_files_names:
        os.remove(f)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Data utilities')
    parser.add_argument('--input_file', type=str, help='Input file', required=True)
    parser.add_argument('--output_file', type=str, help='Output file', default=None)
    parser.add_argument('--cache_dir', type=str, help='Cache directory', default=None)
    parser.add_argument('--max_length', type=int, help='Maximum length', default=512)
    parser.add_argument('--output_dir', type=str, help='Output directory', default=None)

    args = parser.parse_args()
    file_name = args.input_file.split('/')[-1]
    if file_name.startswith('moss'):
        extract_inputs_targets(args.input_file, args.output_file)
        args.input_file = args.output_file
    dataset = load_dataset(args.input_file, args.cache_dir, args.max_length)
    dataset.save_to_disk(args.output_dir)