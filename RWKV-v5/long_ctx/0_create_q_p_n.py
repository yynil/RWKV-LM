import os
import json
from tqdm import tqdm
import regex as re
from rank_bm25 import BM25Okapi
import traceback
INDEX_PATTERN = re.compile('(\[(\d+)\])')
import shutil
from multiprocessing import Pool
import argparse
def handle_text(text :str):
    #replace \n and \t \\ \\n to space
    text = text.replace('\n',' ')
    text = text.replace('\t',' ')
    text = text.replace('\\\\n',' ')
    text = text.replace('\\\\t',' ')

    #tokenize with space and remove the empty string and empty lines
    texts = text.split(' ')
    texts = [t for t in texts if t != '' and t != '\n' and t != '\t' and t != '\r' and t != '\r\n' and t != '\u3000' and t != '\xa0' and t != '\x0b' and t != '\x0c' and t != '\ufeff']

    state = 0
    # iterate tokens and fine the Document , Question and Answer
    index = 0
    current_str = ''
    documents = {}
    question = ''
    for t in texts:
        if state == 0:
            if t == 'Document':
                state = 1
                continue
        elif state == 1:
            m = INDEX_PATTERN.fullmatch(t)
            if m is not None:
                state = 2
                index = int(m.group(2))
                continue
            else:
                #ignore the token between Document and index
                pass
        elif state == 2:
            if t == 'Document':
                state = 1
                documents[index] = current_str
                continue
            elif t == 'Question:':
                state = 3
                documents[index] = current_str
                continue
            else:
                current_str += t + ' '
        elif state == 3:
            question += t + ' '
    answer_index = question.find('Answer:')
    long_answer_index = question.find('Long Answer:',answer_index)
    gold_document_id_index = question.find('Gold Document ID:',long_answer_index)
    answer = question[answer_index + 8:long_answer_index]
    long_answer = question[long_answer_index + 14:gold_document_id_index]
    gold_document_id = question[gold_document_id_index + 17:].strip()
    question = question[:answer_index]
    return documents,question,answer,long_answer,gold_document_id

def process_file(input_file, output_file):
    output_fp = open(output_file,'w',encoding='UTF-8')
    with open(input_file, 'r',encoding='UTF-8') as f:
        lines = f.readlines()
        progress_bar = tqdm(lines,'processing :natural_questions_10_200_docs.jsonl')
        index_no = 0
        valid_lines = 0
        for line in progress_bar:
            index_no += 1
            progress_bar.set_description('processing :natural_questions_10_200_docs.jsonl : %d' % index_no)
            try:
                data = json.loads(line)
                text = data['text']
                documents,question,answer,long_answer,gold_document_id = handle_text(text)
                gold_document_id = int(gold_document_id)
                tokenized_documents = []
                for key in documents.keys():
                    if key == gold_document_id:
                        continue
                    tokenized_documents.append(documents[key].split(' '))
                tokenized_question = question.split(' ')
                doc_scores = BM25Okapi(tokenized_documents).get_scores(tokenized_question)
                #find the most irrelevant document
                min_score = 10000
                min_index = -1
                for index,score in enumerate(doc_scores):
                    if score < min_score:
                        min_score = score
                        min_index = index

                most_irrelevant_document = ' '.join(tokenized_documents[min_index])
                output_data = {
                    'question':question,
                    'answer':answer,
                    'long_answer':long_answer,
                    'positive':documents[gold_document_id],
                    'negative':most_irrelevant_document
                }
                line_of_output = json.dumps(output_data,ensure_ascii=False)
                output_fp.write(line_of_output + '\n')
                valid_lines += 1
            except Exception as e:
                print(e)
                print(index_no)
                continue

        print('valid lines : %d' % valid_lines)

def split_file(input_file, num_splits):
    with open(input_file, 'r', encoding='UTF-8') as f:
        lines = f.readlines()

    split_files = []
    split_size = len(lines) // num_splits

    for i in range(num_splits):
        start = i * split_size
        end = (i + 1) * split_size if i < num_splits - 1 else None
        split_file = f'{input_file}_part_{i}'
        with open(split_file, 'w', encoding='UTF-8') as f:
            f.writelines(lines[start:end])
        split_files.append(split_file)

    return split_files

def parallel_process(input_files, output_files):
    with Pool() as pool:
        pool.starmap(process_file, zip(input_files, output_files))

def concatenate_files(input_files, output_file):
    with open(output_file, 'wb') as outfile:
        for fname in input_files:
            with open(fname, 'rb') as infile:
                shutil.copyfileobj(infile, outfile)

def cleanup_files(files):
    for file in files:
        os.remove(file)

def main(input_file, output_file, num_splits):
    input_files = split_file(input_file, num_splits)
    output_files = [f'output_{i}.jsonl' for i in range(num_splits)]

    parallel_process(input_files, output_files)
    concatenate_files(output_files, output_file)
    cleanup_files(output_files)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='/media/yueyulin/KINGSTON/data/natural_questions_10_200_docs.jsonl', help='Input file path')
    parser.add_argument('--output_file', type=str, default='/media/yueyulin/KINGSTON/data/natural_questions_10_200_docs_q_p_n.jsonl', help='Output file path')
    parser.add_argument('--num_splits', type=int, default=24, help='Number of splits')
    args = parser.parse_args()

    main(args.input_file, args.output_file, args.num_splits)