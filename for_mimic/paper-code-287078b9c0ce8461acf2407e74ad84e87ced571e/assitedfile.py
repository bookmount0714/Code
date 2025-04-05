import json
import os
import random
# ann_path = '../../r2gen/data/iu_xray/annotation.json'
ann_path = '/data1/liuyuxue/Data/mimic_cxr/annotation.json'
fold_path = '/data1/lijunliang/r2gen/data/mimic'
ann = json.loads(open(ann_path, 'r').read())
count = 0

# 随机选取3000个例子
sample_size = 3000
sampled_examples = random.sample(ann['train'], sample_size)

#存储不同字段的数据
field_data = {
    'ids': [],
    'study_ids': [],
    'subject_ids': [],
    'reports': [],
    'image_path': []
}

# 遍历每个例子
for example in ann['train']:
    field_data['ids'].append(example['id'])
    field_data['study_ids'].append(example['study_id'])
    field_data['subject_ids'].append(example['subject_id'])
    field_data['reports'].append(example['report'])
    field_data['image_path'].append(example['image_path'])


# 将不同字段数据保存到各自的 JSON 文件
for field, data in field_data.items():
    output_path = os.path.join(fold_path, f'{field}.json')
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

print("all done")


