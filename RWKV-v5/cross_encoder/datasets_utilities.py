import datasets
from datasets import load_from_disk
def load_cross_encoder_ds_from_disk(directory :str,ctx_len :int):
    train_data = load_from_disk(directory)
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