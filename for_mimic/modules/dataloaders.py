import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from .datasets import IuxrayMultiImageDataset, MimiccxrSingleImageDataset


class R2DataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

            # # 为了防止正则化对SAM分割的影响，将Normalize操作放到visual_extractor中执行
            # self.transform = transforms.Compose([
            #     transforms.Resize(224)])
            #     # transforms.RandomCrop(224),
            #     # transforms.RandomHorizontalFlip()])])

        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
            # self.transform = transforms.Compose([
            #     transforms.Resize((224, 224))])

        if self.dataset_name == 'iu_xray':
            self.dataset = IuxrayMultiImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)
        else:
            self.dataset = MimiccxrSingleImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers
        }
        super().__init__(**self.init_kwargs)

    # @staticmethod
    # def collate_fn(data):
    #     images_id, images, reports_ids, reports_masks, seq_lengths = zip(*data)
    #     images = torch.stack(images, 0)
    #     max_seq_length = max(seq_lengths)
    #
    #     targets = np.zeros((len(reports_ids), max_seq_length), dtype=int)
    #     targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)
    #
    #     for i, report_ids in enumerate(reports_ids):
    #         targets[i, :len(report_ids)] = report_ids
    #
    #     for i, report_masks in enumerate(reports_masks):
    #         targets_masks[i, :len(report_masks)] = report_masks
    #
    #     return images_id, images, torch.LongTensor(targets), torch.FloatTensor(targets_masks)

#mimic用
    @staticmethod
    def collate_fn(data):
        images_id, study_ids, images, reports_ids, reports_masks, seq_lengths = zip(*data)
        images = torch.stack(images, 0)
        max_seq_length = max(seq_lengths)

        targets = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)

        for i, report_ids in enumerate(reports_ids):
            targets[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(reports_masks):
            targets_masks[i, :len(report_masks)] = report_masks

        return images_id, study_ids, images, torch.LongTensor(targets), torch.FloatTensor(targets_masks)
#生成sam用
    # @staticmethod
    # def collate_fn(data):
    #     images_id, image_1, image_2 ,reports_ids, reports_masks, seq_lengths = zip(*data)
    #     # images = torch.stack(images, 0)
    #     image_1 = torch.stack(image_1, 0)
    #     image_2 = torch.stack(image_2, 0)
    #
    #     max_seq_length = max(seq_lengths)
    #
    #     targets = np.zeros((len(reports_ids), max_seq_length), dtype=int)
    #     targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)
    #
    #     for i, report_ids in enumerate(reports_ids):
    #         targets[i, :len(report_ids)] = report_ids
    #
    #     for i, report_masks in enumerate(reports_masks):
    #         targets_masks[i, :len(report_masks)] = report_masks
    #
    #     return images_id, image_1, image_2,torch.LongTensor(targets), torch.FloatTensor(targets_masks)