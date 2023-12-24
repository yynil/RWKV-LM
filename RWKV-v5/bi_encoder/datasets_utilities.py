import datasets
from datasets import load_from_disk
def load_cross_encoder_ds_from_disk(directory :str,ctx_len :int):
    train_data = load_from_disk(directory)
    pad_id = 0
    max_length = ctx_len
    def truncate_and_pad_examples(examples):
        positives = examples['positive']
        negatives = examples['negative']
        queries = examples['query']
        logits_positive = examples['logits_positive']
        logits_negative = examples['logits_negative']
        filtered_positives = []
        filtered_negatives = []
        filtered_queries = []
        filtered_logits_positive = []
        filtered_logits_negative = []
        batch_size = len(positives)
        for i in range(batch_size):
            positive = positives[i]
            negative = negatives[i]
            query = queries[i]
            query_len = len(query)
            postive_len = len(positive)
            negative_len = len(negative)
            if postive_len <= max_length and negative_len <= max_length:
                positive = positive + [pad_id] * (max_length - postive_len)
                negative = negative + [pad_id] * (max_length - negative_len)
                query = query + [pad_id] * (max_length - query_len)
                filtered_positives.append(positive)
                filtered_negatives.append(negative)
                filtered_queries.append(query)
                filtered_logits_positive.append(logits_positive[i])
                filtered_logits_negative.append(logits_negative[i])

        return {"query": filtered_queries, "positive": filtered_positives, "negative": filtered_negatives, "logits_positive": filtered_logits_positive, "logits_negative": filtered_logits_negative}
    train_data = train_data.map(truncate_and_pad_examples, batched=True, num_proc=24, remove_columns=['query','positive', 'negative','logits_positive','logits_negative'])
    return train_data

if __name__ == '__main__':
    dataset_dir = '/media/yueyulin/bigdata/ds/mmarco_chinese/ce_scores/tokenized_ids'
    train_data = load_cross_encoder_ds_from_disk(dataset_dir,256)
    print(train_data['train'][1])
    from torch.utils.data import DataLoader
    import torch
    def collate_fn(examples):
        query = torch.tensor([x['query'] for x in examples], dtype=torch.long)
        positive = torch.tensor([x['positive'] for x in examples], dtype=torch.long)
        negative = torch.tensor([x['negative'] for x in examples], dtype=torch.long)
        logits_positive = torch.tensor([x['logits_positive'] for x in examples], dtype=torch.float)
        logits_negative = torch.tensor([x['logits_negative'] for x in examples], dtype=torch.float)
        return {
            "query": query,
            "positive": positive,
            "negative": negative,
            "logits_positive": logits_positive,
            "logits_negative": logits_negative
        }
    dataloader = DataLoader(train_data['train'], batch_size=4, shuffle=True,collate_fn=collate_fn)
    for batch in dataloader:
        print(batch)
        queries = batch['query']
        actual_queries_len = torch.eq(queries, 1).int().argmax(-1)
        print(actual_queries_len)
        positive = batch['positive']
        actual_positive_len = torch.eq(positive, 1).int().argmax(-1)
        print(actual_positive_len)
        negative = batch['negative']
        actual_negative_len = torch.eq(negative, 1).int().argmax(-1)
        print(actual_negative_len)
        break