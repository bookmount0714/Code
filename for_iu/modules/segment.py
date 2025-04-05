import torch
import torch.nn as nn
import numpy as np
from segment_anything import sam_model_registry,SamAutomaticMaskGenerator,SamPredictor
import sys
sys.path.append("..")
sam_checkpoint = "/data1/lijunliang/sam/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda:0"


def add_anns(anns, image):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    img_float = image.astype(np.float64)
    for ann in sorted_anns:
        m = ann['segmentation']
        indices = np.where(m)
        img_float[indices] *= 0.80
        img_uint8 = np.clip(img_float, 0, 255).astype(np.uint8)
    return img_uint8,img_float


class segment(nn.Module):
    def __init__(self):
        super(segment,self).__init__()
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.model = sam
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )

    def forward(self,image):
        masks = self.mask_generator.generate(image)
        image_uint , image_float = add_anns(masks,image)
        return image_uint

