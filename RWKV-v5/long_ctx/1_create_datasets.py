import argparse
from datasets import load_dataset
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER

def tokenize_examples(examples):
    questions = examples['question']
    answers = examples['answer']
    long_answers = examples['long_answer']
    positives = examples['positive']
    negatives = examples['negative']
    tokenized_examples = []
    import os
    tokenizer_file = os.path.join(os.path.dirname(__file__), 'rwkv_vocab_v20230424.txt')
    tokenizer = TRIE_TOKENIZER(tokenizer_file)
    qt = []
    at = []
    lat = []
    pt = []
    nt = []
    for index,question in enumerate(questions):
        question_tokens = tokenizer.encode(question)
        answer_tokens = tokenizer.encode(answers[index])
        long_answer_tokens = tokenizer.encode(long_answers[index])
        positive_tokens = tokenizer.encode(positives[index])
        negative_tokens = tokenizer.encode(negatives[index])
        qt.append(question_tokens)
        at.append(answer_tokens)
        lat.append(long_answer_tokens)
        pt.append(positive_tokens)
        nt.append(negative_tokens)
    return {
        'question':qt,
        'answer':at,
        'long_answer':lat,
        'positive':pt,
        'negative':nt
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help='Path to the input file')
    parser.add_argument('--output_dir', type=str, help='Path to the output directory')
    args = parser.parse_args()

    input_file = args.input_file
    output_dir = args.output_dir

    dataset = load_dataset('json', data_files=input_file)
    dataset = dataset.map(tokenize_examples,batch_size=24,batched=True, remove_columns=['question','answer','long_answer','positive','negative'],num_proc=24)
    print(dataset)
    print(dataset['train'][0])
    dataset.save_to_disk(output_dir)