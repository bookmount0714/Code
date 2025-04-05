import json
import os
ann_path = '../../r2gen/data/iu_xray/annotation.json'
fold_path = '../../r2gen/data/iu_xray/report_txt'
ann = json.loads(open(ann_path, 'r').read())
count = 0
for example in ann['train']:
    id = example['id']
    report = example['report']
    with open(os.path.join(fold_path,f"{id}.txt"),'w') as f:
        f.write(report)
    count = count +1
    print(id,'_',count ," done")
print("all done")
