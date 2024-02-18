import os 
os.environ["RWKV_JIT_ON"] = '0'
os.environ["RWKV_HEAD_SIZE_A"]='bf16'
import torch
import gc
import sys
src_dir = os.path.dirname(os.path.dirname(__file__))
print(src_dir)
sys.path.append(src_dir)

from src.model_run import RWKV
from src.model_for_sequence_embedding import RwkvForSequenceEmbedding_Run
from src.model_for_classification import RwkvForClassification_Run
import torch
import traceback
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
import chromadb
from torch.amp import autocast

def from_pretrained(model_file,args):
    with torch.no_grad():
        w = torch.load(model_file, map_location='cpu') # load model to CPU first
        gc.collect()
        n_embd = w['emb.weight'].shape[1]
        vocab_size = w['emb.weight'].shape[0]
        dim_att = w['blocks.0.att.key.weight'].shape[0] # note: transposed matrix
        dim_ffn = w['blocks.0.ffn.key.weight'].shape[0] # note: transposed matrix
        n_layer = 0
        keys = list(w.keys())
        version = 4
        for x in keys:
            layer_id = int(x.split('.')[1]) if ('blocks.' in x) else 0
            n_layer = max(n_layer, layer_id+1)
            if 'ln_x' in x:
                version = max(5, version)
            if 'gate.weight' in x:
                version = max(5.1, version)
            if int(version) == 5 and 'att.time_decay' in x:
                n_head = w[x].shape[0]
                if len(w[x].shape) > 1:
                    if w[x].shape[1] > 1:
                        version = max(5.2, version)
        head_size_a = dim_att // n_head
        args.n_embd = n_embd
        args.dim_att = dim_att
        args.dim_ffn = dim_ffn
        args.n_layer = n_layer
        args.version = version
        args.head_size_a = head_size_a
        args.vocab_size = vocab_size
        args.head_qk = 0
        args.dropout = 0
        args.my_pos_emb = 0
        args.pre_ffn = 0
        args.head_size_divisor = 8
        args.ctx_len = 4096
        args.grad_cp = 0
        args.n_head = args.dim_att // args.head_size_a
        model = RWKV(args)
        model.load_state_dict(w,strict=True)
        del w
        gc.collect()
        model = model.bfloat16()
        model = model.cuda()
        model.eval()
        return model

def load_ckpt_and_parse_args(ckpt_file, args):
    try:
        with torch.no_grad():
            w = torch.load(ckpt_file, map_location='cpu') # load model to CPU first
            import gc
            gc.collect()
            
            n_embd = w['emb.weight'].shape[1]
            vocab_size = w['emb.weight'].shape[0]
            dim_att = w['blocks.0.att.key.weight'].shape[0] # note: transposed matrix
            dim_ffn = w['blocks.0.ffn.key.weight'].shape[0] # note: transposed matrix
            n_layer = 0
            keys = list(w.keys())
            version = 4
            for x in keys:
                layer_id = int(x.split('.')[1]) if ('blocks.' in x) else 0
                n_layer = max(n_layer, layer_id+1)
                if 'ln_x' in x:
                    version = max(5, version)
                if 'gate.weight' in x:
                    version = max(5.1, version)
                if int(version) == 5 and 'att.time_decay' in x:
                    n_head = w[x].shape[0]
                    if len(w[x].shape) > 1:
                        if w[x].shape[1] > 1:
                            version = max(5.2, version)
            head_size_a = dim_att // n_head
            args.n_embd = n_embd
            args.dim_att = dim_att
            args.dim_ffn = dim_ffn
            args.n_layer = n_layer
            args.version = version
            args.head_size_a = head_size_a
            args.vocab_size = vocab_size
            args.n_head = args.dim_att // args.head_size_a

            #####Fixed args
            args.head_qk = 0
            args.dropout = 0
            args.my_pos_emb = 0
            args.pre_ffn = 0
            args.head_size_divisor = 8
            
            return w
    except Exception as e:
        traceback.print_exc()
        return None


def inference(model: RwkvForSequenceEmbedding_Run,  tokenizer :TRIE_TOKENIZER,texts :str):
    cls_id = 1
    input_str = texts
    input_ids = tokenizer.encode(input_str)+[cls_id]
    with autocast(device_type='cuda',dtype=torch.bfloat16):
        embeddings = model(torch.tensor([input_ids]).long())
    return embeddings

def inference_rerank(model: RwkvForClassification_Run, template: str, tokenizer :TRIE_TOKENIZER,query :str, document :str):
    cls_id = 1
    input_str = template.format(query=query,document=document)
    input_ids = tokenizer.encode(input_str)+[cls_id]
    with autocast(device_type='cuda',dtype=torch.bfloat16):
        logits = model(torch.tensor([input_ids]).long())
    return logits

def read_text_file(file_name :str):
    full_path = os.path.join(os.path.dirname(__file__),file_name)
    with open(full_path,'r',encoding='UTF-8') as f:
        return f.read().strip()


def read_texts_dir(text_dir :str,chunk_size :int):
    import os
    files = os.listdir(text_dir)
    texts = dict()
    for file in files:
        if file.endswith('.txt'):
            text = read_text_file(os.path.join(text_dir,file))
            offset = 0
            while offset < len(text):
                texts[file+'_'+str(offset)] = text[offset:offset+chunk_size]
                offset += chunk_size
    return texts

from rwkv.utils import PIPELINE, PIPELINE_ARGS
gen_cnt = 0
def my_print(s):
    global gen_cnt
    gen_cnt += 1
    print(s, end='', flush=True)
def generate(pipeline : PIPELINE,ctx :str):
    args = PIPELINE_ARGS(temperature = 1.0, top_p = 0.8, top_k = 100, # top_k = 0 then ignore
                        alpha_frequency = 0.25,
                        alpha_presence = 0.25,
                        alpha_decay = 0.996, # gradually decay the penalty
                        token_ban = [0], # ban the generation of some tokens
                        token_stop = [], # stop generation whenever you see any token here
                        chunk_len = 512) # split input into chunks to save VRAM (shorter -> slower)
    import datetime
    start = datetime.datetime.now()
    with autocast(device_type='cuda',dtype=torch.bfloat16):
        pipeline.generate(ctx, token_count=200, args=args, callback=my_print)
    end = datetime.datetime.now()
    global gen_cnt
    print('\n')
    print(f'tokens/sec: {gen_cnt/(end-start).total_seconds()}')
    gen_cnt = 0


from src.lora_utilities import load_model, set_adapter,set_adapter_layers

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--answer_model', type=str, default='/media/yueyulin/bigdata/models/rwkv5/RWKV-5-World-1B5-v2-20231025-ctx4096.pth')
    parser.add_argument('--type', type=str, default='lora', choices=['full', 'lora'])
    parser.add_argument('--ckpt', type=str, default='/media/yueyulin/bigdata/models/rwkv5/RWKV-5-World-1B5-v2-20231025-ctx4096.pth')
    parser.add_argument('--lora_ckpt', type=str, default='/media/yueyulin/bigdata/models/lora/rwkv1b5/be/trainable_model_0')
    parser.add_argument('--cross_encoder_ckpt', type=str, default='/media/yueyulin/bigdata/models/lora/rwkv1b5/ce_att_ffn/trainable_model_140000/')

    parser.add_argument('--vdb_dir',type=str, default='/home/yueyulin/下载/laws_vdb/')
    parser.add_argument('--chunk_size',type=int, default=1000)
    args = parser.parse_args()
    bi_adapter_name = 'bi_adapter'
    cross_adapter_name = 'cross_adapter'
    args.is_reuse_answer_model = args.answer_model == args.ckpt
    bi_model,ce_model,tokenizer,model = load_model(args,bi_adapter_name=bi_adapter_name,cross_adapter_name=cross_adapter_name)
    if args.is_reuse_answer_model:
        answer_model = model
    else:
        answer_model = from_pretrained(args.answer_model,args)
    pipeline = PIPELINE(answer_model, "rwkv_vocab_v20230424") 
    client = chromadb.PersistentClient(path=args.vdb_dir)
    collection = client.create_collection(
        name="novel_collection",
        metadata={"hnsw:space": "ip"}, # l2 is the defaul
        get_or_create=True
    )
    import datetime
    template = "【问题：{query}\n文档：{document}\n】" 
    ctx = "Instruction:法条：\n{law}\n请根据以上法条回答以下问题。\nInput:问题：{action}\nAssistant:\n"

    while True:
        set_adapter_layers(bi_model,True)
        query = input('query:')
        if query == 'exit':
            break
        start = datetime.datetime.now()
        set_adapter(bi_model,bi_adapter_name)
        end = datetime.datetime.now()
        switch_adapter_time = end-start
        start = datetime.datetime.now()
        query_embedding = inference(bi_model,tokenizer,query).tolist()
        end = datetime.datetime.now()
        bi_encoder_time = end-start
        start = datetime.datetime.now()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=10,
        )
        end = datetime.datetime.now()
        query_vdb_time = end-start
        ids = results['ids'][0]
        distances = results['distances'][0]
        documents = results['documents'][0]
        set_adapter(ce_model,cross_adapter_name)
        re_rank_scores = []
        start = datetime.datetime.now()
        for i in range(len(ids)):
            re_rank_scores.append(inference_rerank(ce_model,template,tokenizer,query,documents[i]))
        end = datetime.datetime.now()
        rerank_time = end-start
        # Sort documents based on re_rank_scores
        sorted_documents = [(score,doc) for score, doc in sorted(zip(re_rank_scores, documents), key=lambda x: x[0] , reverse=True)]
        set_adapter_layers(answer_model,False)
        for i in range(len(sorted_documents)):
            score,doc = sorted_documents[i]
            print(f'score is {score}')
            if score >= 0.7:
                start = datetime.datetime.now()
                input_str = ctx.format(law=doc,action=query)
                print(input_str)
                generate(pipeline,input_str)
                end = datetime.datetime.now()
                generate_time = end-start
                print('\033[91m参考法律条文：',doc,'\033[00m')
                print('\033[91m生成时间：',generate_time,'set adapter time ',switch_adapter_time,'，vdb_query_time is ',query_vdb_time,', rerank time is ',rerank_time,',bi-encode time ',bi_encoder_time, '\033[00m')
