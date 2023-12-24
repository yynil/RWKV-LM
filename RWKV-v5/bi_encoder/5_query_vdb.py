import sys
import os
src_dir = os.path.dirname(os.path.dirname(__file__))
print(src_dir)
sys.path.append(src_dir)

from src.model_for_sequence_embedding import RwkvForSequenceEmbedding_Run
from src.model_for_classification import RwkvForClassification_Run
import torch
import traceback
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
import chromadb

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

def load_full_ckpt_and_parse_args(ckpt_file, args):
    try:
        with torch.no_grad():
            w = torch.load(ckpt_file, map_location='cpu') # load model to CPU first
            import gc
            gc.collect()
            
            n_embd = w['rwkvModel.emb.weight'].shape[1]
            vocab_size = w['rwkvModel.emb.weight'].shape[0]
            dim_att = w['rwkvModel.blocks.0.att.key.weight'].shape[0] # note: transposed matrix
            dim_ffn = w['rwkvModel.blocks.0.ffn.key.weight'].shape[0] # note: transposed matrix
            n_layer = 0
            keys = list(w.keys())
            version = 4
            for x in keys:
                layer_id = int(x.split('.')[2]) if ('blocks.' in x) else 0
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
    from torch.amp import autocast
    with autocast(device_type='cuda',dtype=torch.bfloat16):
        embeddings = model(torch.tensor([input_ids]).long())
    return embeddings

def inference_rerank(model: RwkvForClassification_Run, template: str, tokenizer :TRIE_TOKENIZER,query :str, document :str):
    cls_id = 1
    input_str = template.format(query=query,document=document)
    input_ids = tokenizer.encode(input_str)+[cls_id]
    from torch.amp import autocast
    with autocast(device_type='cuda',dtype=torch.bfloat16):
        logits = model(torch.tensor([input_ids]).long())
    print(logits)
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

def load_model(args):
    w = load_ckpt_and_parse_args(args.ckpt, args)
    from src.model_run import RWKV
    model = RWKV(args)
    model.eval()
    inform = model.load_state_dict(w,strict=False)
    del w 
    #list lora_ckpt as directory
    import os
    if not os.path.isdir(args.lora_ckpt):
        print('lora_ckpt should be a directory')
        exit(-1)
    files = os.listdir(args.lora_ckpt)
    ckpt_file = None
    lora_config = None
    for file in files:
        if file.endswith('.pth'):
            ckpt_file = os.path.join(args.lora_ckpt,file)
        elif file.endswith('.json'):
            lora_config = os.path.join(args.lora_ckpt,file)
        
        if ckpt_file is not None and lora_config is not None:
            break
    print('load ckpt from ',ckpt_file,' and config from ',lora_config)
    w = torch.load(ckpt_file, map_location='cpu')
    import json
    from peft import LoraConfig,TaskType,inject_adapter_in_model
    with open(lora_config,'r') as f:
        lora_obj = json.load(f)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            lora_alpha=lora_obj['lora_alpha'],
            lora_dropout=0,
            r=lora_obj['r'],
            bias=lora_obj['bias'],
            target_modules=lora_obj['target_modules'],)
        print(lora_config)
    model = inject_adapter_in_model(lora_config,model)
    print(model)
    model = RwkvForSequenceEmbedding_Run(model,chunk_size=1024)
    print(model)
    inform = model.load_state_dict(w,strict=False)
    print(inform)
    import os
    tokenizer = TRIE_TOKENIZER(os.path.dirname(os.path.abspath(__file__)) + '/rwkv_vocab_v20230424.txt' )
    model = model.bfloat16()
    model = model.to('cuda')
    model.eval()
    
    print(inform)

    from src.model_run import RWKV
    ce_model = RWKV(args)
    w = load_ckpt_and_parse_args(args.ckpt, args)
    inform = ce_model.load_state_dict(w,strict=False)
    del w
    files = os.listdir(args.cross_encoder_ckpt)
    ckpt_file = None
    cross_encoder_config = None
    for file in files:
        if file.endswith('.pth'):
            ckpt_file = os.path.join(args.cross_encoder_ckpt,file)
        elif file.endswith('.json'):
            cross_encoder_config = os.path.join(args.cross_encoder_ckpt,file)
        
        if ckpt_file is not None and cross_encoder_config is not None:
            break
    print('load cross encoder ckpt from ',ckpt_file,' and config from ',cross_encoder_config)
    w = torch.load(ckpt_file, map_location='cpu')
    num_labels = w['score.weight'].shape[0]
    from peft import inject_adapter_in_model
    with open(cross_encoder_config,'r') as f:
        cross_encoder_obj = json.load(f)
        cross_encoder_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            lora_alpha=cross_encoder_obj['lora_alpha'],
            lora_dropout=0,
            r=cross_encoder_obj['r'],
            bias=cross_encoder_obj['bias'],
            target_modules=cross_encoder_obj['target_modules'],)
        print(cross_encoder_config)
    ce_model = inject_adapter_in_model(cross_encoder_config,ce_model)
    
    ce_model = RwkvForClassification_Run(ce_model, num_labels,chunk_size=1024)
    inform = ce_model.load_state_dict(w,strict=False)
    print(inform)
    ce_model = ce_model.bfloat16()
    ce_model = ce_model.to('cuda')
    ce_model.eval()
    return model,ce_model,tokenizer

from chromadb import Documents, EmbeddingFunction, Embeddings

class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self,model,tokenizer) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for doc in input:
            embeddings.append(inference(self.model,self.tokenizer,doc))
        return embeddings

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='lora', choices=['full', 'lora'])
    parser.add_argument('--ckpt', type=str, default='/media/yueyulin/bigdata/models/rwkv5/RWKV-5-World-1B5-v2-20231025-ctx4096.pth')
    parser.add_argument('--lora_ckpt', type=str, default='/media/yueyulin/bigdata/models/lora/rwkv1b5/be/trainable_model_0')
    parser.add_argument('--cross_encoder_ckpt', type=str, default='/media/yueyulin/bigdata/models/lora/rwkv1b5/ce_att_ffn/trainable_model_140000/')

    parser.add_argument('--vdb_dir',type=str, default='/home/yueyulin/下载/laws_vdb')
    parser.add_argument('--chunk_size',type=int, default=1000)
    args = parser.parse_args()

    model,ce_model,tokenizer = load_model(args)
    print(model)
    print(ce_model)
    print(tokenizer)

    client = chromadb.PersistentClient(path=args.vdb_dir)
    collection = client.create_collection(
        name="novel_collection",
        metadata={"hnsw:space": "ip"}, # l2 is the defaul
        get_or_create=True,
        embedding_function=MyEmbeddingFunction(model,tokenizer)
    )

    template = "【问题：{query}\n文档：{document}\n】" 
    while True:
        query = input('query:')
        if query == 'exit':
            break
        query_embedding = inference(model,tokenizer,query).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=10,
        )
        ids = results['ids'][0]
        distances = results['distances'][0]
        documents = results['documents'][0]

        re_rank_scores = []
        for i in range(len(ids)):
            re_rank_scores.append(inference_rerank(ce_model,template,tokenizer,query,documents[i]))
        print(re_rank_scores)
        # Sort documents based on re_rank_scores
        sorted_documents = [(score,doc) for score, doc in sorted(zip(re_rank_scores, documents), key=lambda x: x[0] , reverse=True)]
        for i in range(len(sorted_documents)):
            score,doc = sorted_documents[i]
            if score > 0.5:
                print(score,doc)