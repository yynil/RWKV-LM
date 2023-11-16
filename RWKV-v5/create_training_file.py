import json

from tqdm import tqdm

def parse_json(filename,outputfile):
    with open(filename, 'r') as f:
        data = json.load(f)
    
    output_data = []
    for item in tqdm(data, desc="Processing data"):
        dataType = item.get('dataType')
        title = item.get('title')
        content = item.get('content')
        
        # print(f"User: 根据以下内容，撰写文章标题。{content}\nAssistant: 类型：{dataType}\n标题：{title}")
        output_data.append(f"User: 根据以下内容，撰写文章标题。{content}\nAssistant: 类型：{dataType}\n标题：{title}")
        # output_data.append({'dataType': dataType, 'title': title, 'content': content})
    
    with open(outputfile, 'w') as f:
        json.dump(output_data, f,ensure_ascii=False)

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default='data.json', help='json file')
    parser.add_argument('--output', type=str, default='output.txt', help='output file')
    args = parser.parse_args()
    
    parse_json(args.filename,args.output)