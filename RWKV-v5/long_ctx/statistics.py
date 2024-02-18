from datasets import load_from_disk
if __name__ == '__main__':
    ds_dir = '/media/yueyulin/KINGSTON/data/natural_questions_10_200_docs_q_p_n_tokenized_ds'
    ds = load_from_disk(ds_dir)
    print(ds)

    import numpy as np
    from tqdm import tqdm
    progress = tqdm(total=len(ds['train']))
    pos = []
    neg = []
    for data in ds['train']:
        question = data['question']
        positive = data['positive']
        negative = data['negative']
        q_p_len = len(question) + len(positive)+2
        q_n_len = len(question) + len(negative)+2
        pos.append(q_p_len)
        neg.append(q_n_len)
        progress.update(1)
    pos_arr = np.array(pos)
    neg_arr = np.array(neg)
    
    np.save('/media/yueyulin/KINGSTON/data/natural_questions_10_200_docs_q_p_n_tokenized_ds/pos_arr.npy',pos_arr)
    np.save('/media/yueyulin/KINGSTON/data/natural_questions_10_200_docs_q_p_n_tokenized_ds/neg_arr.npy',neg_arr)

    