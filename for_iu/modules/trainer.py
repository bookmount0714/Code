import os
from abc import abstractmethod

import time
import torch
import pandas as pd
from numpy import inf
import hdbscan
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch.nn.functional as F
import copy




import logging
from . import globals_ele

device = "cuda:0"
def get_logger(dirname,  filename, verbosity=1):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "%(message)s"
    )
    logger = logging.getLogger(filename)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(dirname+filename, "a")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def zero_pad_after_zero(tensor):
    result = torch.zeros_like(tensor)
    for i in range(tensor.shape[0]):
        zero_found = False
        for j in range(tensor.shape[1]):
            if tensor[i, j] == 0 and not zero_found:
                zero_found = True
            if zero_found:
                result[i, j] = 0
            else:
                result[i, j] = tensor[i, j]
    return result

class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args):
        self.args = args

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        log_dir = self.args.log_dir
        if self.args.dataset_name == 'iu_xray':
            log_file = 'iu_xray.log'
        elif self.args.dataset_name == 'mimic_cxr':
            log_file = 'mimic_cxr.log'
        else:
            log_file = 'cov_ctr.log'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logger = get_logger(dirname=log_dir, filename=log_file)
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)
            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            logger.info(log)
            self._record_best(log)

            # print logged informations to the screen
            for key, value in log.items():
                print('\t{:15s}: {}'.format(str(key), value))


            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    print("Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                        self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    print("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                        self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
        self._print_best()
        self._print_best_to_file()

    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time
        self.best_recorder['test']['time'] = crt_time
        self.best_recorder['val']['seed'] = self.args.seed
        self.best_recorder['test']['seed'] = self.args.seed
        self.best_recorder['val']['best_model_from'] = 'val'
        self.best_recorder['test']['best_model_from'] = 'test'

        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)
        record_path = os.path.join(self.args.record_dir, self.args.dataset_name+'.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        record_table = pd.concat([record_table, pd.DataFrame(self.best_recorder['val'], index=[0])], ignore_index=True)
        record_table = pd.concat([record_table, pd.DataFrame(self.best_recorder['test'], index=[0])], ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
            self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                            self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)

    def _print_best(self):
        print('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            print('\t{:15s}: {}'.format(str(key), value))

        print('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['test'].items():
            print('\t{:15s}: {}'.format(str(key), value))


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def _train_epoch(self, epoch):
        #迭代开始前对reports_tensor进行更新
        # globals_ele.reports_tensor = torch.from_numpy(self.model.sentence_bert.encode(globals_ele.reports))
        # #一个迭代开始前对报告进行分类
        # embedding = self.model.reducer.fit_transform(globals_ele.reports_tensor)
        # clusterer = hdbscan.HDBSCAN(min_cluster_size=15, metric='euclidean')
        # cluster_labels = clusterer.fit_predict(embedding)
        #
        # # 处理噪声点，将 -1 替换为聚类数目
        # num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        # cluster_labels = np.where(cluster_labels == -1, num_clusters, cluster_labels)
        #
        # # 将标签转换为独热编码
        # encoder = OneHotEncoder(sparse=False)
        # # report_labels记录报告分类的独热编码，每次迭代前进行更新
        # report_labels = encoder.fit_transform(cluster_labels.reshape(-1, 1))
        report_labels = np.load('/data1/lijunliang/r2gen/data/iu_xray/report_labels.npy')


        train_loss = 0
        self.model.train()
        for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.train_dataloader):
            images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), reports_masks.to(
                self.device)
            #从report_labels取出该批次报告对应分类:batch_report_labels
            batch_report_labels = []
            for id in images_id:
                try:
                    index = globals_ele.ids.index(id)
                    # 从 report_labels 中取出对应的张量并添加到列表
                    batch_report_labels.append(report_labels[index])
                except ValueError:
                    pass

            batch_report_labels = [torch.from_numpy(arr) for arr in batch_report_labels]
            batch_report_labels = torch.stack(batch_report_labels).to(dtype = torch.float32).to(device)

            output = self.model(images, reports_ids,batch_report_labels,images_id ,mode='train')

            #增加生成句子与实际句子在语义上的损失
            # 对第三维进行排序并获取下标
            idx = torch.zeros((output.shape[0],output.shape[1]))
            for i in range(output.shape[0]):
                for j in range(output.shape[1]):
                    _, index = torch.max(output[i, j, :], dim=0)
                    idx[i, j] = index

            idx = zero_pad_after_zero(idx)
            reports = self.model.tokenizer.decode_batch(idx.cpu().numpy())
            print(reports[0])

            ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].detach().cpu().numpy())#实际报告
            reports_tensor = torch.from_numpy(self.model.sentence_bert.encode(reports)).to(device)
            ground_truths = torch.from_numpy(self.model.sentence_bert.encode(ground_truths)).to(device)
            similarities = self.model.sentence_bert.similarity(reports_tensor, ground_truths)
            s = F.relu(similarities)
            diagonal_elements = torch.diag(s)
            loss_meaning =  -torch.sum(torch.log(diagonal_elements))#语义上的损失
            print(loss_meaning)

            loss = self.criterion(output, reports_ids, reports_masks)
            loss_sum = loss + globals_ele.loss_selected_report + globals_ele.loss_image_report_384 + globals_ele.loss_memory_sam + globals_ele.loss_classify + loss_meaning
            # loss_sum =  loss + globals_ele.loss_selected_report + globals_ele.loss_image_report_384 + globals_ele.loss_memory_sam + loss_meaning
            train_loss += loss.item()
            self.optimizer.zero_grad()
            # loss.backward()
            loss_sum.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            print('the batch number is: {}, current loss: {}'.format(batch_idx, loss))
        print('train loss: {}'.format(train_loss / len(self.train_dataloader)))
        log = {}

        self.model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.val_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                output = self.model(images,images_id = images_id, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            log.update(**{'val_' + k: v for k, v in val_met.items()})

        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                output = self.model(images,images_id = images_id,mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                print(reports)
                test_res.extend(reports)
                test_gts.extend(ground_truths)
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})

        self.lr_scheduler.step()

        return log
