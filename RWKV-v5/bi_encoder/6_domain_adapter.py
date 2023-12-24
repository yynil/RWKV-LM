import torch
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
def split_texts(text_dir :str,ctx_len :int = 500):
    import os

    files =  os.listdir(text_dir)
    files = [os.path.join(text_dir, file) for file in files]
    files = [file for file in files if os.path.isfile(file)]
    files = [file for file in files if file.endswith(".txt")]
    results = {}
    for f in files:
        with open(f, "r", encoding="utf-8") as fin:
            text = fin.read()
            texts = text.split("\n")
            grouped_texts = []
            grouped_text = ""
            for t in texts:
                if len(grouped_text) + len(t) < ctx_len:
                    grouped_text += t
                else:
                    grouped_texts.append(grouped_text)
                    grouped_text = t
            grouped_texts.append(grouped_text)
            file_name_without_ext = os.path.splitext(os.path.basename(f))[0]
            for i, t in enumerate(grouped_texts):
                results[f"{f}_{i}"] = f"《{file_name_without_ext}》第{i}段：{t}"
    return results

def init_doc2query_model():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    model_name = 'doc2query/msmarco-chinese-mt5-base-v1'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda")
    return tokenizer, model

def create_queries(para, tokenizer, model):
    input_ids = tokenizer.encode(para, return_tensors='pt').to("cuda")
    with torch.no_grad():
        # Here we use top_k / top_k random sampling. It generates more diverse queries, but of lower quality
        sampling_outputs = model.generate(
            input_ids=input_ids,
            max_length=64,
            do_sample=True,
            top_p=0.95,
            top_k=10, 
            num_return_sequences=5
            )
        
        # Here we use Beam-search. It generates better quality queries, but with less diversity
        beam_outputs = model.generate(
            input_ids=input_ids, 
            max_length=64, 
            num_beams=5, 
            no_repeat_ngram_size=2, 
            num_return_sequences=5, 
            early_stopping=True
        )
    return sampling_outputs, beam_outputs

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_dir", type=str, default="/home/yueyulin/下载/法律/")
    parser.add_argument("--ctx_len", type=int, default=500)
    args = parser.parse_args()
    results = split_texts(args.text_dir, args.ctx_len)
    tokenizer, model = init_doc2query_model()
    for k, v in results.items():
        print(k)
        print(v)
        sampling_outputs, beam_outputs = create_queries(v, tokenizer=tokenizer, model=model)
        print("\nBeam Outputs:")
        for i in range(len(beam_outputs)):
            print(tokenizer.decode(beam_outputs[i], skip_special_tokens=True))
        print("\nSampling Outputs:")
        for i in range(len(sampling_outputs)):
            print(tokenizer.decode(sampling_outputs[i], skip_special_tokens=True))
        

    