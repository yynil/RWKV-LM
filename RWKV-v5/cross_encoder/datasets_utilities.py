import datasets
from datasets import load_from_disk
def load_cross_encoder_ds_from_disk(directory :str,ctx_len :int):
    train_data = load_from_disk(directory)
    features = train_data['train'].features
    if 'positive' in features and 'negative' in features:
        pad_id = 0
        max_length = ctx_len
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

            inputs = filtered_positives + filtered_negatives
            targets = [1] * len(filtered_positives) + [0] * len(filtered_negatives)
            return {"idx": inputs, "targets": targets}
        train_data = train_data.map(truncate_and_pad_examples, batched=True, num_proc=24, remove_columns=['positive', 'negative'])
        return train_data
    elif 'input_ids' in features and 'labels' in features:
        def rename_input_ids_and_labels(examples):
            return {'idx':examples['input_ids'],'targets':examples['labels']}
        train_data = train_data.map(rename_input_ids_and_labels,batched=True,num_proc=24,remove_columns=['input_ids','labels'])
        return train_data
    

if __name__ == '__main__':
    # old_format_ds = "/media/yueyulin/bigdata/ds/mmarco_chinese/rwkv_tokenized_ids"
    # ds = load_cross_encoder_ds_from_disk(old_format_ds, 256)
    # print(ds)

    new_format_ds = "/media/yueyulin/KINGSTON/data/natural_questions_10_200_docs_q_p_n_tokenized_ds_padded_16384_cls_1_sep_2_truncate_False_pad_0"
    ds = load_cross_encoder_ds_from_disk(new_format_ds, 256)
    print(ds)