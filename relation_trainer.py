import argparse
import os, sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.training_utils import TrainingProgress, timeSince, LearningRateScheduler
from model.Relation_Predictor import Relationship_Predictor
import utils.array_tool as at
from model.faster_rcnn_vgg16 import FasterRCNNVGG16
from model.faster_rcnn_resnet import FasterRCNN_ResNet
from VMRD_dataset import VMRD_dataset, Preprocess
from collections import namedtuple
from torchnet.meter import ConfusionMeter, AverageValueMeter
from tqdm import tqdm
import numpy as np
from img_type import tensor_to_PIL
from PIL import Image
from torchvision import transforms
from ipdb import set_trace
import random as rd
from model.utils.combine_feature import get_combined_feature

IMAGE_HEIGHT_AFTER_CROPPED = 240
IMAGE_WIDTH_AFTER_CROPPED = 320
NUM_OF_CLASS = 3

global_config = {
    'model_dir': os.path.dirname(os.path.abspath(__file__)) + '/model/',
    'progress_dir': os.path.dirname(os.path.abspath(__file__)) + '/progress/relation/',
    'data_dir': os.path.dirname(os.path.abspath(__file__)) + '/datasets/VMRD/',
}

LossTuple = namedtuple('LossTuple',
                        ['total_loss'])

class Relation_Trainer:
    def __init__(self, args):
        self.conf = args
        self.check_cuda()
        # Check with eval tools
        self.tp = TrainingProgress(global_config['progress_dir'], self.conf.model + self.conf.opt_header + '-progress',
                                   data_key_list=['epoch_loss', 'test_loss', 'training_loss', 'mAP'])
        self.load_net()
        self.load_data(args)
        self.create_data_loader()
        self.prepare_training()  # Optimizer/Loss function
    def check_cuda(self):
        if torch.cuda.is_available():
            torch.cuda.set_device(self.conf.dev)  # default 0
            print("Use CUDA,device=", torch.cuda.current_device())
            self.device = 'cuda:' + str(self.conf.dev)
        else:
            self.device = 'cpu'
        at.gpu_dev = self.device

    def load_net(self):
        """
        Load network weight or initiate a new model
        """
        pretrained = not self.conf.res  # if not loading weight
        self.net = Relationship_Predictor(n_rel_class=NUM_OF_CLASS, use_extractor=True, pretrained=pretrained, spatial_feature=self.conf.use_spatial_feature)
        self.net = self.net.to(self.device)

    def load_data(self,args):  # batch_files, data_conf, cuda_dev):
        """
        Load Training/Testing data
        """

        pp_train = Preprocess(normalize_means=[0.485, 0.456, 0.406], normalize_stds=[0.229, 0.224, 0.225], p_hflip=0.5)
        pp_test = Preprocess(normalize_means=[0.485, 0.456, 0.406], normalize_stds=[0.229, 0.224, 0.225], p_hflip=0)
        self.train_set = VMRD_dataset(data_dir=global_config['data_dir'], dataset_name=args.trd, split='trainval',
                                       preprocess=pp_train)
        self.test_set = VMRD_dataset(data_dir=global_config['data_dir'], dataset_name=args.trd, split='test',
                                      preprocess=pp_test)
        print('Data loader: Training set: ', len(self.train_set), ' Testing set: ', len(self.test_set))

    def create_data_loader(self):

        self.train_loader = DataLoader(dataset=self.train_set, batch_size=self.conf.tr_bat, shuffle=False,
                                       pin_memory=True, num_workers=8)
        self.test_loader = DataLoader(dataset=self.test_set, batch_size=self.conf.ts_bat, shuffle=False,
                                      pin_memory=True, num_workers=12)
        # self.valid_loader = DataLoader(dataset=self.val_set, batch_size=self.conf.ts_bat, shuffle=False,
        #                                pin_memory=True, num_workers=8)


    def prepare_training(self):
        """
        Load/Create Meta data
        Load/Restore Current Progress
        Set training parameters, Init optimizer
        """
        self.tp.add_meta({'conf': self.conf, 'global_conf': global_config})
        if self.conf.res:
            self.restore_progress()  # set optimizer and lr_scheduler
        else:
            self.epoch = 1
            self.set_optimizer()
            self.set_lr_scheduler()
        self.init_meters()
        # self.set_loss_func()

    def train(self):
        """
        while loss<target loss
            forward
            backward
            record loss
            if loop_n % RECORD_N:
                summary & save_progress
        """
        time_start = time.time()
        self.net.train()
        try:
            while self.epoch < self.conf.max_epoch:
                self.epoch_loss = 0
                self.reset_meters()
                for step, (img_name, img, bbox, label, index, relation, scale, _) in tqdm(enumerate(self.train_loader)): #bbox shape is [1, 2, 4] #label is [1,n] label_attr is[1,m,n]
                    if bbox.size(1) < 2:
                        continue

                    self.optimizer.zero_grad()
                    img, bbox, label, index, relation, scale = \
                        img.to(self.conf.dev), bbox.to(self.conf.dev), label.to(self.conf.dev), index.to(self.conf.dev),\
                        relation.to(self.conf.dev), at.scalar(scale)
                    loss = self.train_forward_net(img_name, img, bbox, label, index, relation, scale)
                    #print("loss {0}".format(loss.total_loss))
                    loss.total_loss.backward()
                    self.optimizer.step()
                    self.update_meters(loss)
                    self.epoch_loss += loss.total_loss.detach().cpu().numpy() * img.size(0)

                    #if step == 3500:
                    #    break
                    # if step % 200 == 0:
                    #     print('Step=', step)
                # ['epoch_loss', 'test_loss', 'training_loss']
                self.epoch_loss = self.epoch_loss / len(self.train_loader.dataset)
                # self.valid_loss = self.test(use_validation=True, display=True)
                self.tp.record_data({'epoch_loss': self.epoch_loss})  # 'validation_loss': self.valid_loss})
                self.lr_scheduler.step({'loss': self.epoch_loss, 'epoch': self.epoch})  # , 'torch': self.valid_loss})
                if self.epoch % self.conf.se == 0:
                    print(timeSince(time_start), ': Trainer Summary Epoch=', self.epoch)
                    self.summary()
                self.epoch += 1
            print(timeSince(time_start), ': Trainer Summary Epoch=', self.epoch)
            self.summary(save_optim=True)  # for resume training
        except KeyboardInterrupt:
            save = input('Save Current Progress ? y for yes: ')
            if 'y' in save:
                print('Saving Progress...')
                self.save_progress(save_optim=True, display=True)
    def train_forward_net(self, img_names, imgs, bboxes, labels, indexs, gt_relations, scale):
        n = bboxes.shape[0]
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        # Since batch size is one, convert variables to singular form
        bboxes = bboxes[0]
        gt_relations = gt_relations[0]
        _, _, H, W = imgs.shape  # Tensor type (C,H,W)

        feature = self.net.extractor(imgs)
        combined_features, rel_flip = get_combined_feature(feature, bboxes, use_spatial_feature=self.conf.use_spatial_feature)
        relation_scores = []
        for combined_feature in combined_features:
            relation_score = self.net(*combined_feature)
            relation_scores.append(relation_score)
        relation_scores = torch.stack(relation_scores, dim=1).squeeze(0)

        ##to see if need flip
        for n, flip in enumerate(rel_flip):
            if flip: #if flip
                if gt_relations[n] == 1:
                    gt_relations[n] = 2
                elif gt_relations[n] == 2:
                    gt_relations[n] = 1
        gt_relations = at.to_tensor(gt_relations).long().cuda()
        #self.rel_cm.add(relation_scores, gt_relations.data)
        relation_loss = nn.CrossEntropyLoss()(relation_scores, gt_relations) ##(onehot tensor, not onehot long tensor)
        return LossTuple(total_loss=relation_loss)

    # Meters
    def init_meters(self):
        self.rel_cm = ConfusionMeter(3)  # num of class(including background)
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}  # average loss

    def update_meters(self, losses):
        loss_d = {k: at.scalar(v) for k, v in losses._asdict().items()}
        for key, meter in self.meters.items():
            meter.add(loss_d[key])

    def reset_meters(self):
        for key, meter in self.meters.items():
            meter.reset()
        self.rel_cm.reset()

    def get_meter_data(self):
        return {k: v.value()[0] for k, v in self.meters.items()}
    def set_optimizer(self):
        """
        Set optimizer parameters
        """
        if self.conf.optim == 'SGD':
            self.optimizer = getattr(optim, 'SGD')(filter(lambda p: p.requires_grad, self.net.parameters()),
                                                   lr=self.conf.lr_init, momentum=0.9, nesterov=True,
                                                   weight_decay=self.conf.w_decay)  # default SGD
        else:
            self.optimizer = getattr(optim, self.conf.optim)(
                filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.conf.lr_init,
                weight_decay=self.conf.w_decay)  # default SGD
        if self.conf.res:
            if self.tp.get_meta('optim') == self.conf.optim:
                if 'optim_state' in self.tp.meta_dict.keys():
                    self.optimizer.load_state_dict(self.tp.get_meta('optim_state'))
                    print('Optimizer Internal State Restored')
    def set_lr_scheduler(self, restore_dict=None):
        self.lr_scheduler = LearningRateScheduler(self.conf.lrs, self.optimizer, self.conf.lr_rates,
                                                  self.conf.lr_epochs, self.conf.lr_loss, self.conf.lr_init, None)
        if restore_dict is not None:
            self.lr_scheduler.step(restore_dict)
        pass

    def summary(self, save_optim=False):  # Do the tests
        """
        Record the training and testing loss/time/accuracy
        """
        # train_loss = self.test(use_training=True, display=True)
        # test_loss = self.test(display=True)

        test_map = self.eval_relation_predictor()['map']
        # valid_loss = self.test(use_validation=True) #Already record in training loop

        self.tp.record_data({'mAP': test_map}, display=True)
        self.tp.add_meta(
            {'saved_epoch': self.epoch, 'epoch_loss': self.epoch_loss})  # , 'validation_loss': self.valid_loss})
        self.save_progress(display=True, save_optim=save_optim)

    def save_progress(self, display=False, save_optim=False):
        """
        Save training weight/progress/meta data
        """
        self.net = self.net.to('cpu')
        self.tp.add_meta(
            {'net_weight': self.net.state_dict(), 'optim': self.conf.optim})
        if self.conf.save_opt or save_optim:
            print('Saving Optimizer Sate')
            self.tp.add_meta({'optim_state': self.optimizer.state_dict()})
        self.tp.save_progress(self.epoch)

        self.net = self.net.to(self.device)
        if display:
            print('Progress Saved, current epoch=', self.epoch)

    def restore_progress(self):
        """
        Restore training weight/progress/meta data
        Restore self.epoch,optimizer parameters
        """
        self.tp.restore_progress(self.conf.tps)

        self.net = self.net.to('cpu')
        self.net.load_state_dict(self.tp.get_meta('net_weight'))

        self.net = self.net.to(self.device)
        # restore all the meta data and variables
        self.epoch = self.tp.get_meta('saved_epoch')
        self.epoch_loss = self.tp.get_meta('epoch_loss')
        # self.valid_loss = self.tp.get_meta('validation_loss')
        print('Restore Progress,epoch=', self.epoch, ' epoch loss=', self.epoch_loss)
        self.set_optimizer()
        self.set_lr_scheduler(restore_dict={'epoch': self.epoch})
        self.epoch += 1
    def eval_relation_predictor(self):
        pred_relations, pred_relation_scores = [], []
        gt_relations = []
        for ii, (img_name, imgs_, bboxes_, _, _, gt_relations_, _, _) \
                in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):  # gt_label is numpy (1,N物體)
            if bboxes_.size(1) < 2:
                continue
            imgs_, bboxes_ = imgs_.to(self.conf.dev), bboxes_.to(self.conf.dev)
            pred_relations_, pred_relation_scores_ = self.net.predict(imgs_, bboxes_[0]) ##numpy form and not one hot

            # print("predict shape {0}".format(pred_bboxes_.shape[0]))
            gt_relations.append(gt_relations_[0].numpy())
            pred_relations.append(pred_relations_)
            pred_relation_scores.append(pred_relation_scores_)
            assert gt_relations[-1].shape == pred_relations[-1].shape, "gt:{0}, pre:{1}".format(gt_relations[-1].shape,pred_relations[-1].shape)

        # print(len(pred_bboxes), len(gt_bboxes))
        accu_arr = np.zeros(shape=(NUM_OF_CLASS, 2), dtype=int) #0 right, 1 total
        for pred_relations_, gt_relations_ in zip(pred_relations, gt_relations):
            for i in range(len(pred_relations_)):
                if pred_relations_[i] == gt_relations_[i]:
                    accu_arr[gt_relations_[i], 0] += 1#加gound truth 那個
                accu_arr[gt_relations_[i], 1] += 1

        ap = np.empty(shape=NUM_OF_CLASS, dtype=float)
        for ii in range(NUM_OF_CLASS):
            if accu_arr[ii, 1] == 0: ##non has been tested
                ap[ii] = np.nan
            else:
                ap[ii] = float(accu_arr[ii,0] / accu_arr[ii,1])
        result = {'ap': ap, 'map': np.nanmean(ap)}
        print("ap: {0}".format(result['ap']))
        print('map: {0}'.format(result['map']))
        return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', help='Model file name', type=str, default='rel_detect_model')
    parser.add_argument('-net_model', help='net Model(vgg or resnet)', type=str, default='ResNet')  ##new
    parser.add_argument('-opt_header', help='Additional weight/progress file header', type=str, default='')
    parser.add_argument('-trd', help='Training dataset', type=str, default='')  ##new
    parser.add_argument('-ted', help='Testing dataset', type=str, default='')  ## VOC2012 can't be the test dataset
    parser.add_argument('-dev', help='CUDA Device Number, used when GPU is available', type=int, default=0)
    parser.add_argument('-tr', help='Training Data Ratio (Validation ration is 1-tr)', type=float, default=0.95)

    # Learning rate scheduler setting
    parser.add_argument('-lrs', help='Learning rate scheduler mode', type=str, default='epoch')
    parser.add_argument('-lr_init', help='Initial Learning rate for Torch lr Schedulers', type=float, default=1e-3)
    parser.add_argument('-lr_rates', help='Learning Rates for epochs.', nargs='+', type=float,
                        default=[1e-3, 1e-4, 5e-5])
    parser.add_argument('-lr_epochs', help='Epochs for Learning Rate control.', nargs='+', type=int, default=[7, 12])
    parser.add_argument('-lr_loss', help='Loss targets for Learning Rate control.', nargs='+', type=float, default=None)

    parser.add_argument('-optim', help='Overwrite the optimizer, default=SGD.',
                        choices=['RMSprop', 'SGD', 'Adadelta', 'Adam'], default='SGD')
    parser.add_argument('-w_decay', help='Weight decay parameter for Optimizer. Ex: 1e-5', type=float, default=5e-4)

    parser.add_argument('-se', help='Save progress and do summary every ? epoch', type=int, default=1)
    parser.add_argument('--res', help='Restore progress and resume the training progress',
                        action='store_true', default=False)
    parser.add_argument('-tps', help='Restore Training Progress index(step)', type=int, default=8)
    parser.add_argument('--save_opt', help='Save optimizer State Dict (Take some time !)?', action='store_true',
                        default=False)

    parser.add_argument('--test', help='Display Testing Result only!', action='store_true', default=False)

    parser.add_argument('-tr_bat', help='Training Batch Size', type=int, default=1)
    parser.add_argument('-ts_bat', help='Testing Batch Size', type=int, default=1)

    parser.add_argument('-max_epoch', help='Max Epoch for training', type=int, default=5)  ##15 orginal
    parser.add_argument('-min_loss', help='Minimum Loss for training', type=float, default=1e-5)
    parser.add_argument('--init_fin', help='Weight Initialization by the fan-in size', action='store_true',
                        default=False)
    parser.add_argument('--para', help='Parallel Training', action='store_true', default=False)
    parser.add_argument('--use_spatial_feature', help='Whether use spatial feature', action='store_true', default=False)

    args = parser.parse_args()
    trainer = Relation_Trainer(args)
    print("training dataset : {0}       testing dataset : {1}".format(args.trd, args.ted))
    trainer.train()
if __name__ == '__main__':
       main()
