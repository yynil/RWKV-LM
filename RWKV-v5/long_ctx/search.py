import torch
import numpy as np

if __name__ == '__main__':
    npy_file = '/media/yueyulin/KINGSTON/data/natural_questions_10_200_docs_q_p_n_tokenized_ds/neg_arr.npy'
    arr = np.load(npy_file)
    x_len = torch.tensor([1,2,3,4,5,6],dtype=torch.float,requires_grad=True)
    inputs = torch.tensor(arr,dtype=torch.long)
    max_len = 32*1024
    import torch.optim as optim
    optimizer = optim.Adam([x_len],lr=0.01)
    for i in range(100):
        optimizer.zero_grad()
        pad_size = torch.zeros_like(inputs)
        for j in range(len(inputs)):
            mask = x_len > inputs[j]  # 创建一个掩码，标记x_len中大于inputs[j]的元素
            greater_elements = torch.masked_select(x_len, mask)  # 使用掩码选择x_len中的元素
            if greater_elements.numel() > 0:  # 如果存在大于inputs[j]的元素
                cut_off = torch.min(greater_elements)  # 获取这些元素中的最小值
            else:  # 如果不存在大于inputs[j]的元素
                cut_off = max_len  # 使用inputs[j]作为cut_off
            pad_size[j] = cut_off - inputs[j]
        #loss is to minimize the sum of pad_size
        loss = torch.sum(pad_size)
        loss.backward()
        optimizer.step()
        print(x_len)
        print(loss)
        break
