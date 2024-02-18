# 更新日期 2024-2-18

1. Cross Encoder的格式修改成： query_ids + sep_id + document_ids + cls_id + paddings， 取cls_id为logits的值
2. Bi Encoder的格式修改为: text_ids + cls_id， 取cls_id的值为embeddings， loss 为positie embeddings和negatives embeddings的dot product的分差接近cross encoder的分差
3. 修改了Bi/Cross/SFT训练代码，删掉不必要的to(device)操作
4. 增加了对moss+firefly微调数据集的sft过程
5. src/lora_utilities.py现在增加了三个lora交叉使用的调用。
6. 几个Lora的分享地址：

    6.1 Bi-Encoder：链接: https://pan.baidu.com/s/1tvIW0pWMCPrjrjeIkhDPaA?pwd=kxjb 提取码: kxjb 
--来自百度网盘超级会员v6的分享

    6.2 Cross-Encoder：链接: https://pan.baidu.com/s/1yZzoKCtxUBJUvuTMtVq-RQ?pwd=5qmm 提取码: 5qmm 
--来自百度网盘超级会员v6的分享

    6.3 SFT：链接: https://pan.baidu.com/s/1L1r4MKTwd7d3hM1vdZS2ew?pwd=e3jy 提取码: e3jy 
--来自百度网盘超级会员v6的分享

7. 所有的Lora均从RWKV5-7B正式版开始训练