import argparse
from datasets import load_from_disk
import os

def pad_and_truncate_examples(examples, max_length, sep_token_id, cls_token_id, is_truncate=True, pad_token_id=0):
    questions = examples['question']
    # answers = examples['answer']
    # long_answers = examples['long_answer']
    positives = examples['positive']
    negatives = examples['negative']
    inputs = []
    labels = []
    for index, question in enumerate(questions):
        question_tokens = question
        positive_tokens = positives[index]
        negative_tokens = negatives[index]
        input_positive_tokens = question_tokens + [sep_token_id] + positive_tokens + [cls_token_id]
        input_negative_tokens = question_tokens + [sep_token_id] + negative_tokens + [cls_token_id]
        skip = False
        if len(input_positive_tokens) > max_length:
            if is_truncate:
                input_positive_tokens = input_positive_tokens[:max_length]
            else:
                skip = True
        if not skip:
            if len(input_negative_tokens) > max_length:
                if is_truncate:
                    input_negative_tokens = input_negative_tokens[:max_length]
                else:
                    skip = True
        if not skip:
            input_positive_tokens = input_positive_tokens + [pad_token_id] * (max_length - len(input_positive_tokens))
            inputs.append(input_positive_tokens)
            labels.append(1)
            input_negative_tokens = input_negative_tokens + [pad_token_id] * (max_length - len(input_negative_tokens))
            inputs.append(input_negative_tokens)
            labels.append(0)
    return {'input_ids': inputs, 'labels': labels}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_dir', type=str, help='Directory of the dataset')
    parser.add_argument('--max_length', type=int, help='Maximum length of the input')
    parser.add_argument('--sep_token_id', type=int, help='ID of the separator token')
    parser.add_argument('--cls_token_id', type=int, help='ID of the classification token')
    parser.add_argument('--is_truncate', action='store_true', help='是否截断数据')
    parser.add_argument('--pad_token_id', type=int, help='ID of the padding token')
    args = parser.parse_args()

    ds_dir = args.ds_dir
    max_length = args.max_length
    sep_token_id = args.sep_token_id
    cls_token_id = args.cls_token_id
    is_truncate = args.is_truncate
    pad_token_id = args.pad_token_id

    parent_dir = os.path.dirname(ds_dir)
    ds = load_from_disk(ds_dir)
    
    output_ds_dir = f'{parent_dir}/natural_questions_10_200_docs_q_p_n_tokenized_ds_padded_{max_length}_cls_{cls_token_id}_sep_{sep_token_id}_truncate_{is_truncate}_pad_{pad_token_id}'

    from functools import partial
    pad_and_truncate_examples_fn = partial(pad_and_truncate_examples, max_length=max_length, sep_token_id=sep_token_id, cls_token_id=cls_token_id, is_truncate=is_truncate, pad_token_id=pad_token_id)
    ds = ds.map(pad_and_truncate_examples_fn, batched=True, num_proc=24, batch_size=24, remove_columns=ds['train'].features.keys())
    print(ds['train'][0])
    print(ds)
    ds.save_to_disk(output_ds_dir)
