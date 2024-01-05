from datasets import load_from_disk
if __name__ == '__main__':
    ds_dir = '/media/yueyulin/KINGSTON/data/natural_questions_10_200_docs_q_p_n_tokenized_ds'
    ds = load_from_disk(ds_dir)
    print(ds)

    p_len = []
    n_len = []
    for data in ds['train']:
        question = data['question']
        positive = data['positive']
        negative = data['negative']
        q_p_len = len(question) + len(positive)+2
        q_n_len = len(question) + len(negative)+2
        p_len.append(q_p_len)
        n_len.append(q_n_len)

    import numpy as np
    print(f'positive length: mean {np.mean(p_len)}, max {np.max(p_len)}, min {np.min(p_len)}')
    print(f'negative length: mean {np.mean(n_len)}, max {np.max(n_len)}, min {np.min(n_len)}')