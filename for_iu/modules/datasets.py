import os
import json

import numpy
import torch
from PIL import Image
from torch.utils.data import Dataset
import cv2
import sys
from segment_anything import sam_model_registry,SamAutomaticMaskGenerator,SamPredictor
sys.path.append("..")
sam_checkpoint = "../sam/sam_vit_h_4b8939.pth"
class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())

        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

    def __len__(self):
        return len(self.examples)


# class IuxrayMultiImageDataset(BaseDataset):
#     def __getitem__(self, idx):
#         example = self.examples[idx]
#         image_id = example['id']
#         image_path = example['image_path']
#         image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
#         image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
#         #为确保image分割效果 使用cv读取图像数据
#         if self.transform is not None:
#             image_1 = torch.from_numpy(numpy.array(self.transform(image_1)))
#             image_2 = torch.from_numpy(numpy.array(self.transform(image_2)))
#         image = torch.stack((image_1, image_2), 0)
#         report_ids = example['ids']
#         report_masks = example['mask']
#         seq_length = len(report_ids)
#         sample = (image_id, image, report_ids, report_masks, seq_length)
#
#         return sample


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        image_1_sam = Image.open(os.path.join(self.image_dir, image_path[0].replace('0.png','0_sam.png'))).convert('RGB')
        image_2_sam = Image.open(os.path.join(self.image_dir, image_path[1].replace('0.png','0_sam.png'))).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
            image_1_sam = self.transform(image_1_sam)
            image_2_sam = self.transform(image_2_sam)
        image = torch.stack((image_1, image_2,image_1_sam,image_2_sam), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)

        return sample

class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample
