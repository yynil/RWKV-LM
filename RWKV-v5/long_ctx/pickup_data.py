import argparse
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import jieba
if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='/media/yueyulin/KINGSTON/data/zh.json',help='Path to the input file')
    parser.add_argument('--num_samples',type=int, default=10,help='Number of samples to pick')
    parser.add_argument('--max_ctx_len',type=int, default=1024, help='Maximum context length')
    parser.add_argument('--prob',type=float,default=0.8, help='Probability of picking a sample')
    parser.add_argument('--output_dir',default='/home/yueyulin/training/corpus', type=str, help='Path to the output directory')
    args = parser.parse_args()

    input_file = args.input_file
    output_dir = args.output_dir
    num_samples = args.num_samples
    max_ctx_len = args.max_ctx_len
    prob = args.prob

    import os
    import json
    import random
    from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
    tokenizer_file = os.path.join(os.path.dirname(__file__), 'rwkv_vocab_v20230424.txt')
    tokenizer = TRIE_TOKENIZER(tokenizer_file)

    #make dir if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file_name = os.path.basename(input_file)
    output_file_name = f'{output_file_name}_sample_{num_samples}_max_ctx_len_{max_ctx_len}_prob_{prob}.jsonl'
    output_file = os.path.join(output_dir,output_file_name)

    with open(input_file,'r',encoding='UTF-8') as input_fp, open(output_file,'w',encoding='UTF-8') as output_fp:
        valid_lines = 0
        progress_bar = tqdm(total=num_samples)
        for index,line in enumerate(input_fp):
            if index % 10000 == 0:
                print(index)
            try:
                data = json.loads(line)
                query = data['query']
                positive = data['pos'][0]
                negatives = data['neg']
                #pick a most irrelevant negative document
                texts = [list(jieba.cut(doc)) for doc in negatives]
                bm25 = BM25Okapi(texts)
                input_query = list(jieba.cut(query))
                scores = bm25.get_scores(input_query)
                ranked_documents = sorted(zip(negatives, scores), key=lambda x: x[1], reverse=True)
                most_irrelevant_document = ranked_documents[-1][0]

                query_len = len(tokenizer.encode(query))
                positive_len = len(tokenizer.encode(positive))
                negative_len = len(tokenizer.encode(most_irrelevant_document))
                if query_len * 2 + positive_len + negative_len + 4 > 2 * max_ctx_len:
                    continue
                if random.random() > prob:
                    continue
                output_data = {
                    'query':query,
                    'positive':positive,
                    'negative':most_irrelevant_document
                }
                line_of_output = json.dumps(output_data,ensure_ascii=False)
                output_fp.write(line_of_output + '\n')
                valid_lines += 1
                progress_bar.update(1)
                if valid_lines >= num_samples:
                    break

            except Exception as e:
                print(e)
                print(index)
                continue

        print('valid lines : %d' % valid_lines)


    