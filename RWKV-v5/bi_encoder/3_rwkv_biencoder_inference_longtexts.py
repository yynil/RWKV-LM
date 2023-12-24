import sys
import os
src_dir = os.path.dirname(os.path.dirname(__file__))
print(src_dir)
sys.path.append(src_dir)

from src.model_for_sequence_embedding import RwkvForSequenceEmbedding_Run
import torch
import traceback
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
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
    print(embeddings)
    return embeddings

def read_text_file(file_name :str):
    full_path = os.path.join(os.path.dirname(__file__),file_name)
    with open(full_path,'r',encoding='UTF-8') as f:
        return f.read().strip()

if __name__ == '__main__':
    beijing = read_text_file('beijing.txt')
    tianjin = read_text_file('tianjin.txt')

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='lora', choices=['full', 'lora'])
    parser.add_argument('--ckpt', type=str, default='/media/yueyulin/bigdata/models/rwkv5/RWKV-5-World-1B5-v2-20231025-ctx4096.pth')
    parser.add_argument('--lora_ckpt', type=str, default="/media/yueyulin/bigdata/models/lora/rwkv1b5/be/trainable_model_0")
    args = parser.parse_args()
    if args.type == 'full':
        w  = load_full_ckpt_and_parse_args(args.ckpt, args)
        from src.model_run import RWKV
        model = RWKV(args)
        model = RwkvForSequenceEmbedding_Run(model)
        inform = model.load_state_dict(w,strict=False)
        print(model)
        print(inform)
        query = "北京有几个火车站"
        document_positive = beijing
        document_negative = tianjin

        
        import os
        tokenizer = TRIE_TOKENIZER(os.path.dirname(os.path.abspath(__file__)) + '/rwkv_vocab_v20230424.txt' )
        model = model.bfloat16()
        model = model.to('cuda')
        model.eval()

    elif args.type == 'lora':
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
        model = RwkvForSequenceEmbedding_Run(model)
        print(model)
        inform = model.load_state_dict(w,strict=False)
        import os
        tokenizer = TRIE_TOKENIZER(os.path.dirname(os.path.abspath(__file__)) + '/rwkv_vocab_v20230424.txt' )
        model = model.bfloat16()
        model = model.to('cuda')
        model.eval()
        
        print(inform)
        query = "北京有几个火车站"
        document_positive = beijing
        document_negative = tianjin

        query_embeddings = inference(model, tokenizer, query)
        query_negative = inference(model, tokenizer, document_negative)
        query_positive = inference(model, tokenizer, document_positive)
        from sentence_transformers.util import pairwise_dot_score as score_fn
        score_positive = score_fn(query_embeddings, query_positive)
        score_negative = score_fn(query_embeddings, query_negative)
        print(f'{query} vs positive',score_positive)
        print(f'{query} vs negative',score_negative)