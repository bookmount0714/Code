import os

import numpy
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
from segment_anything import sam_model_registry,SamAutomaticMaskGenerator,SamPredictor
from PIL import Image
from torchvision import transforms
from modules.dataloaders import R2DataLoader
from main import parse_agrs
from modules.tokenizers import Tokenizer



device = "cuda:0"

args = parse_agrs()
tokenizer = Tokenizer(args)
# path = "../r2gen/data/iu_xray/images/"
suffix = "_sam"
sam_checkpoint = "../sam/sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,
)
# 本文件对图片进行预处理(分割处理),因为在训练过程中分割太过耗时


# val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
# test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)





def add_anns(anns, image):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    img_float = image.astype(np.float64)
    for ann in sorted_anns:
        m = ann['segmentation']
        indices = np.where(m)
        img_float[indices] *= 0.75
        img_uint8 = np.clip(img_float, 0, 255).astype(np.uint8)
    return img_uint8,img_float
count = 1


train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True)
print("traindata begin")
for batch_idx, (image_path,images_id, images,reports_ids, reports_masks) in enumerate(train_dataloader):

    image_path = os.path.join('/data1/liuyuxue/Data/mimic_cxr/images',image_path[0][0])

    image = images[0]

    # 获取原图片的目录、文件名和扩展名
    directory, original_filename = os.path.split(image_path)
    filename_without_ext, ext = os.path.splitext(original_filename)

    #新文件名
    new_filename = f"{filename_without_ext}_sam{ext}"
    new_image_path = os.path.join(directory, new_filename)

    if not os.path.exists(new_image_path):
        mask_1 = mask_generator.generate(numpy.array(image))
        image_1_sam, _ = add_anns(mask_1, numpy.array(image))
        image_1_sam = Image.fromarray(image_1_sam)
        image_1_sam.save(new_image_path)
        print(new_image_path)
    print(count," done")
    count = count + 1
print("traindata done")
count = 1
print("valdata begin")

val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
for batch_idx, (image_path,images_id, images,reports_ids, reports_masks) in enumerate(val_dataloader):

    image_path = os.path.join('/data1/liuyuxue/Data/mimic_cxr/images',image_path[0][0])

    image = images[0]

    # 获取原图片的目录、文件名和扩展名
    directory, original_filename = os.path.split(image_path)
    filename_without_ext, ext = os.path.splitext(original_filename)

    #新文件名
    new_filename = f"{filename_without_ext}_sam{ext}"
    new_image_path = os.path.join(directory, new_filename)

    if not os.path.exists(new_image_path):
        mask_1 = mask_generator.generate(numpy.array(image))
        image_1_sam, _ = add_anns(mask_1, numpy.array(image))
        image_1_sam = Image.fromarray(image_1_sam)
        image_1_sam.save(new_image_path)
    print(count," done")
    count = count + 1
print("valdata done")
count = 1
print("testdata begin")

test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)
for batch_idx, (image_path,images_id, images,reports_ids, reports_masks) in enumerate(test_dataloader):

    image_path = os.path.join('/data1/liuyuxue/Data/mimic_cxr/images',image_path[0][0])

    image = images[0]

    # 获取原图片的目录、文件名和扩展名
    directory, original_filename = os.path.split(image_path)
    filename_without_ext, ext = os.path.splitext(original_filename)

    #新文件名
    new_filename = f"{filename_without_ext}_sam{ext}"
    new_image_path = os.path.join(directory, new_filename)

    if not os.path.exists(new_image_path):
        mask_1 = mask_generator.generate(numpy.array(image))
        image_1_sam, _ = add_anns(mask_1, numpy.array(image))
        image_1_sam = Image.fromarray(image_1_sam)
        image_1_sam.save(new_image_path)
    print(count," done")
    count = count + 1
print("testdata done")

# for batch_idx, (images_id, image_1,image_2, reports_ids, reports_masks) in enumerate(train_dataloader):
#     image_1 = image_1[0,:]
#     image_2 = image_2[0,:]
#     output_dir = '../r2gen/data/iu_xray/images/' + images_id[0]
#
#     output_path_0 = os.path.join(output_dir, '0'+suffix+'.png')
#     output_path_1 = os.path.join(output_dir, '1'+suffix+'.png')
#
#
#     if not os.path.exists(output_path_0):
#         mask_1 = mask_generator.generate(numpy.array(image_1))
#         image_1_sam, _ = add_anns(mask_1, numpy.array(image_1))
#         image_1_sam = Image.fromarray(image_1_sam)
#         image_1_sam.save(output_path_0)
#
#     if not os.path.exists(output_path_1):
#         mask_2 = mask_generator.generate(numpy.array(image_2))
#         image_2_sam, _ = add_anns(mask_2, numpy.array(image_2))
#         image_2_sam = Image.fromarray(image_2_sam)
#         image_2_sam.save(output_path_1)
#
#     print(count," done")
#     count = count + 1
#
# print("traindata done")
# count = 1
# for batch_idx, (images_id, image_1, image_2, reports_ids, reports_masks) in enumerate(val_dataloader):
#     image_1 = image_1[0, :]
#     image_2 = image_2[0, :]
#     output_dir = '../r2gen/data/iu_xray/images/' + images_id[0]
#
#     output_path_0 = os.path.join(output_dir, '0'+suffix+'.png')
#     output_path_1 = os.path.join(output_dir, '1'+suffix+'.png')
#
#     if not os.path.exists(output_path_0):
#         mask_1 = mask_generator.generate(numpy.array(image_1))
#         image_1_sam, _ = add_anns(mask_1, numpy.array(image_1))
#         image_1_sam = Image.fromarray(image_1_sam)
#         image_1_sam.save(output_path_0)
#
#     if not os.path.exists(output_path_1):
#         mask_2 = mask_generator.generate(numpy.array(image_2))
#         image_2_sam, _ = add_anns(mask_2, numpy.array(image_2))
#         image_2_sam = Image.fromarray(image_2_sam)
#         image_2_sam.save(output_path_1)
#
#     print(count , " done")
#     count = count + 1
# print("valdata done")
#
# count = 1
# for batch_idx, (images_id, image_1, image_2, reports_ids, reports_masks) in enumerate(test_dataloader):
#     image_1 = image_1[0, :]
#     image_2 = image_2[0, :]
#     output_dir = '../r2gen/data/iu_xray/images/' + images_id[0]
#
#     output_path_0 = os.path.join(output_dir, '0'+suffix+'.png')
#     output_path_1 = os.path.join(output_dir, '1'+suffix+'.png')
#
#     if not os.path.exists(output_path_0):
#         mask_1 = mask_generator.generate(numpy.array(image_1))
#         image_1_sam, _ = add_anns(mask_1, numpy.array(image_1))
#         image_1_sam = Image.fromarray(image_1_sam)
#         image_1_sam.save(output_path_0)
#
#     if not os.path.exists(output_path_1):
#         mask_2 = mask_generator.generate(numpy.array(image_2))
#         image_2_sam, _ = add_anns(mask_2, numpy.array(image_2))
#         image_2_sam = Image.fromarray(image_2_sam)
#         image_2_sam.save(output_path_1)
#
#     print(count , " done")
#     count = count + 1
# print("testdata done")