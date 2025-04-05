import json
import torch
import numpy as np

loss_memory_sam = 0
loss_selected_report = 0
loss_image_report_384 = 0
loss_classify = 0
radgraph_iu_data = 0

reports_tensor_file = '../r2gen/data/iu_xray/reports_tensor.npy'
reports_file = '../r2gen/data/iu_xray/reports.json'
ids_file = '../r2gen/data/iu_xray/ids.json'
iu_report_radgraph = '../r2gen/data/iu_xray/iu_radgraph_report.json'

ids = json.loads(open(ids_file, 'r').read())
reports = json.loads(open(reports_file, 'r').read())
reports_tensor = torch.from_numpy(np.load(reports_tensor_file))
report_dict = {id: reports_tensor[i] for i, id in enumerate(ids)}
