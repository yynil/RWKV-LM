
import os 
os.environ["RWKV_JIT_ON"] = '0'
os.environ["RWKV_HEAD_SIZE_A"]='64'
from src.model_run import RWKV
import torch
import gc
import datetime
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
        return model

import torch.nn.functional as F
import numpy as np

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, default="/Volumes/WINDOWS10/models/RWKV-5-World-1B5-v2-20231025-ctx4096.pth", help='model file')
    args = parser.parse_args()
    model = from_pretrained(args.model_file,args)
    print(model)
    device = 'mps'
    model = model.to(device)
    model.eval()
    from rwkv.utils import PIPELINE, PIPELINE_ARGS
    pipeline = PIPELINE(model, "rwkv_vocab_v20230424") # 20B_tokenizer.json is in https://github.com/BlinkDL/ChatRWKV

    #ctx = "\n繁城之下讲述了一个以明代为背景的故事。明万历年间，蠹县发生了一起连环凶杀案。蠹县刚正不阿的冷捕头被长杆穿过，钉死在麦田中央的稻草人之上，凶手还在尸体上附上字条：”吾将一道以贯之“。冷捕头的徒弟曲三更发誓要为师傅报仇，找到真凶。然而随着调查的深入，蠹县连续发生离奇命案，凶手都在凶案现场附上不同的字条。"
    ctx = "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."
    print(ctx, end='')

    gen_cnt = 0

    def my_print(s):
        global gen_cnt
        gen_cnt += 1
        print(s, end='', flush=True)

    # For alpha_frequency and alpha_presence, see "Frequency and presence penalties":
    # https://platform.openai.com/docs/api-reference/parameter-details

    args = PIPELINE_ARGS(temperature = 1.0, top_p = 0.7, top_k = 100, # top_k = 0 then ignore
                        alpha_frequency = 0.25,
                        alpha_presence = 0.25,
                        alpha_decay = 0.996, # gradually decay the penalty
                        token_ban = [0], # ban the generation of some tokens
                        token_stop = [], # stop generation whenever you see any token here
                        chunk_len = 256) # split input into chunks to save VRAM (shorter -> slower)
    import datetime
    start = datetime.datetime.now()
    strOutput = pipeline.generate(ctx, token_count=200, args=args, callback=my_print)
    end = datetime.datetime.now()
    print(f'tokens/sec: {gen_cnt/(end-start).total_seconds()}')
    print('\n')
    print(strOutput)
