import numpy
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
from segment_anything import sam_model_registry,SamAutomaticMaskGenerator,SamPredictor
from PIL import Image
from torchvision import transforms
# from mia.MIA import MIA
#
# mia = MIA(
#     d_model=512,N=2,
#     d_inner=2048,n_head=8, d_k=64, d_v=64,
#     dropout=0.1)
#
# image = np.random.rand(16, 98, 512).astype(np.float32)
# words = np.random.rand(16, 59, 512).astype(np.float32)
#
# image , words  = mia( torch.from_numpy(image), torch.from_numpy(words))
#
# print(image.shape)



sys.path.append("..")
def show_anns(anns,image):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    # ax = plt.gca()
    # ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    img_float = image.astype(np.float64)
    for ann in sorted_anns:
        m = ann['segmentation']
        indices = np.where(m)
        # color_mask = np.concatenate([np.random.random(3), [0.35]])
        # img[m] = color_mask #color_mask是一个四个元素的，前三个元素代表三通道的数值，最后一个数代表透明度  这行代码表示将img中对应m中为true位置的像素着色
        img_float[indices] *= 0.80
        img_uint8 = np.clip(img_float, 0, 255).astype(np.uint8)
    # plt.imshow(img_uint8)
    # plt.show()
    return img_uint8,img_float


#fortensor
# def show_anns(anns,image):
#     if len(anns) == 0:
#         return
#     sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
#
#     for ann in sorted_anns:
#         m = ann['segmentation']
#         indices = np.where(m)
#         image[indices] *= 0.85
#         # img_uint8 = np.clip(img_float, 0, 255).astype(np.uint8)
#     # plt.imshow(img_uint8)
#     # plt.show()
#     return image

# def show_anns(anns):
#     if len(anns) == 0:
#         return
#     sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
#     ax = plt.gca()
#     ax.set_autoscale_on(False)
#
#     img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
#     img[:,:,3] = 0
#     for ann in sorted_anns:
#         m = ann['segmentation']
#         color_mask = np.concatenate([np.random.random(3), [0.35]])
#         img[m] = color_mask
#     ax.imshow(img)

# transform = transforms.Compose([
#                 transforms.Resize(256),
#                 transforms.RandomCrop(224),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.485, 0.456, 0.406),
#                                      (0.229, 0.224, 0.225))])
# transform = transforms.Compose([
#                 transforms.Resize(256),
#                 transforms.RandomCrop(224),
#                 transforms.RandomHorizontalFlip()])

img = Image.open('/data1/lijunliang/r2gen/data/iu_xray/images/CXR1007_IM-0008/0.png').convert('RGB')
test = numpy.array(img)
tensor = torch.from_numpy(test)


img = cv2.imread('/data1/lijunliang/r2gen/data/iu_xray/images/CXR1007_IM-0008/0.png')
image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# image_tensor = transform(img).permute(1,2,0)
# image_tensor = image_tensor.mul(255).add_(0.5).clamp_(0,255)
# image_tensor_numpy = image_tensor.numpy()



# plt是可以绘制tensor类型的
plt.imshow(tensor)

plt.axis('off')
plt.show()

plt.clf()
sam_checkpoint = "../sam/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda:0"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
# mask_generator = SamAutomaticMaskGenerator(sam)
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,
)

masks = mask_generator.generate(numpy.array(tensor))
print(len(masks))
print(masks[0].keys())


image , image_float= show_anns(masks,numpy.array(tensor))
# image = show_anns(masks,image_tensor)
plt.imshow(image)
plt.show()

# plt.figure(figsize=(20,20))
# plt.imshow(image)
# show_anns(masks)
# plt.axis('off')
# plt.show()




