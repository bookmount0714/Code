import json
import torch
import numpy as np


loss_memory_sam = 0
loss_selected_report = 0
loss_image_report_384 = 0
loss_classify = 0
radgraph_iu_data = 0

radgraph_mimic_data = 0

# reports_tensor_file = '../r2gen/data/iu_xray/reports_tensor.npy'
# reports_file = '../r2gen/data/iu_xray/reports.json'
# ids_file = '../r2gen/data/iu_xray/ids.json'
# iu_report_radgraph = '../r2gen/data/iu_xray/iu_radgraph_report.json'
#
# ids = json.loads(open(ids_file, 'r').read())
# reports = json.loads(open(reports_file, 'r').read())
# reports_tensor = torch.from_numpy(np.load(reports_tensor_file))
# report_dict = {id: reports_tensor[i] for i, id in enumerate(ids)}


#for mimic
reports_file = '/data1/lijunliang/r2gen/data/mimic/reports.json'
reports_tensor_file = '/data1/lijunliang/r2gen/data/mimic/reports_tensor.npy'
study_ids_file = '/data1/lijunliang/r2gen/data/mimic/study_ids.json'
ids_file = '/data1/lijunliang/r2gen/data/mimic/ids.json'
mimic_report_radgraph = '/data1/liuyuxue/code_scsr/SCSR/radgraph_mimic.json'
study_ids_rad_file = '/data1/lijunliang/r2gen/data/mimic/study_ids_rad.json'
reports_rad_file = '/data1/lijunliang/r2gen/data/mimic/reports_rad.json'
reports_tensor_rad_file = '/data1/lijunliang/r2gen/data/mimic/reports_tensor_rad.npy'
study_ids_mimic_file = '/data1/lijunliang/r2gen/data/mimic/study_ids_mimic.json'
reports_mimic_file = '/data1/lijunliang/r2gen/data/mimic/reports_mimic.json'
reports_tensor_mimic_file = '../../r2gen/data/mimic/reports_tensor_mimic.npy'

findings = '/data1/lijunliang/r2gen/data/mimic/findings.json'
keys = '/data1/lijunliang/r2gen/data/mimic/keys.json'
findings_tensor = '/data1/lijunliang/r2gen/data/mimic/findings.npy'
complete_keys = '/data1/lijunliang/r2gen/data/mimic/complete_keys.json'



id_file = '/data1/lijunliang/r2gen/data/mimic/assited/study_ids.json'
path = '/data1/lijunliang/r2gen/data/mimic/assited/image_complete_path.json'
report_file = '/data1/lijunliang/r2gen/data/mimic/assited/reports.json'
report_tensor_file = '/data1/lijunliang/r2gen/data/mimic/assited/reports_tensor.npy'

#带rad尾缀的是从ann中提取出来，去掉重复之后的数据，15万条左右
study_ids_ann = json.loads(open(id_file,'r').read())
reports_ann = json.loads(open(report_file, 'r').read())
reports_tensor_ann = torch.from_numpy(np.load(report_tensor_file))
path = json.loads(open(path, 'r').read())
report_dict = {id: reports_tensor_ann[i] for i, id in enumerate(study_ids_ann)}
# ids = json.loads(open(ids_file, 'r').read())


#radgraph中提取出来的
study_ids = json.loads(open(keys, 'r').read())
reports = json.loads(open(findings, 'r').read())
reports_tensor = torch.from_numpy(np.load(findings_tensor))


#带mimic尾缀的是从radgraph中提取出来的,8万多条
# study_ids_mimic = json.loads(open(study_ids_mimic_file, 'r').read())
# reports_mimic = json.loads(open(reports_mimic_file, 'r').read())
# reports_tensor_mimic = json.loads(open(reports_tensor_mimic_file, 'r').read())








