import sys
import os
src_dir = os.path.dirname(os.path.dirname(__file__))
print(src_dir)
sys.path.append(src_dir)

import torch
import traceback
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
from src.model_for_sequence_embedding import RwkvForSequenceEmbedding_Run
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


def inference_be(model: RwkvForSequenceEmbedding_Run, tokenizer :TRIE_TOKENIZER,query :str):
    cls_id = 1
    input_ids = tokenizer.encode(query)+[cls_id]
    from torch.amp import autocast
    with autocast(device_type='cuda',dtype=torch.bfloat16):
        logits = model(torch.tensor([input_ids]).long())
    return logits
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='lora', choices=['full', 'lora'])
    parser.add_argument('--ckpt', type=str, default='/media/yueyulin/bigdata/models/rwkv5/RWKV-5-World-1B5-v2-20231025-ctx4096.pth')
    parser.add_argument('--lora_ckpt', type=str, default='/media/yueyulin/bigdata/models/lora/rwkv1b5/ce_att_ffn/trainable_model_140000/')
    args = parser.parse_args()

    w = load_ckpt_and_parse_args(args.ckpt, args)
    from src.model_run import RWKV
    model = RWKV(args)
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
    model = RwkvForSequenceEmbedding_Run(model,delete_head=True)
    print(model)
    inform = model.load_state_dict(w,strict=False)
    import os
    tokenizer = TRIE_TOKENIZER(os.path.dirname(os.path.abspath(__file__)) + '/rwkv_vocab_v20230424.txt' )
    model = model.bfloat16()
    model = model.to('cuda')    
    model.eval()
    query = "中华人民共和国是什么时候成立的？"
    document_positive = "中华人民共和国（the People's Republic of China），简称“中国”，成立于1949年10月1日 [1]，位于亚洲东部，太平洋西岸 [2]，是工人阶级领导的、以工农联盟为基础的人民民主专政的社会主义国家 [3]，以五星红旗为国旗 [4]、《义勇军进行曲》为国歌 [5]，国徽中间是五星照耀下的天安门，周围是谷穗和齿轮 [6] [170]，通用语言文字是普通话和规范汉字 [7]，首都北京 [8]，是一个以汉族为主体、56个民族共同组成的统一的多民族国家。中国陆地面积约960万平方千米，东部和南部大陆海岸线1.8万多千米，海域总面积约473万平方千米 [2]。海域分布有大小岛屿7600多个，其中台湾岛最大，面积35798平方千米 [2]。中国同14国接壤，与8国海上相邻。省级行政区划为23个省、5个自治区、4个直辖市、2个特别行政区。 [2]"
    # document_negative = "河北省，简称“冀”，是中华人民共和国省级行政区，省会石家庄，位于北纬36°05′-42°40′，东经113°27′-119°50′之间，环抱首都北京市，东与天津市毗连并紧傍渤海，东南部、南部衔山东省、河南省，西倚太行山与山西为邻，西北部、北部与内蒙古自治区交界，东北部与辽宁省接壤，总面积18.88万平方千米。 [1-2]河北省下辖11个地级市，共有49个市辖区、21个县级市、91个县、6个自治县。 [4-5]截至2022年末，河北省常住人口为7420万人。 [3] [132]"
    document_negative = "美国原为印第安人聚居地。15世纪末，西班牙、荷兰、法国、英国等开始向北美移民。到1773年，英已建立13个殖民地。1775年，爆发独立战争。1776年7月4日，通过《独立宣言》，正式宣布建立美利坚合众国。1787年，制定联邦宪法，南方为蓄奴州，北方为自由州。1865年南北战争结束，美国开始全面实行资本主义。建国后的领土几乎扩张了10倍。 [1]"
    document_negative = "abcdefg,hijklmn"
    query_embeddings = inference_be(model, tokenizer, query)
    negative_embeddings = inference_be(model, tokenizer, document_negative)
    positive_embeddings = inference_be(model, tokenizer, document_positive)
    from sentence_transformers.util import pairwise_dot_score as score_fn
    score_positive = score_fn(query_embeddings, positive_embeddings)
    score_negative = score_fn(query_embeddings, negative_embeddings)
    print(f'positive score: {score_positive}, negative score: {score_negative}')