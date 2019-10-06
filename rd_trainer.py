import argparse
import os, sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.training_utils import TrainingProgress, timeSince, LearningRateScheduler
from utils.eval_tool import eval_detection_voc
from model.utils.target_creators import AnchorTargetCreator, ProposalTargetCreator
import utils.array_tool as at
from model.faster_rcnn_vgg import FasterRCNN_VGG
from model.faster_rcnn_resnet import FasterRCNN_ResNet
from RD_dataset import RD_dataset, Preprocess
from collections import namedtuple
from torchnet.meter import ConfusionMeter, AverageValueMeter
from tqdm import tqdm
from utils.accu_eval import accu_eval
from torchvision.transforms import ToPILImage
from PIL import Image
from plot_tool import display_image
from plot_tool import vis_bbox, vis_image
from img_type import tensor_to_PIL
import matplotlib.pyplot as plt
import numpy as np

from ipdb import set_trace
#import pydevd

#print(torch.__version__)
# pydevd.settrace('140.113.216.12', port=9988, stdoutToServer=True, stderrToServer=True)

#NUM_OF_CATERGORIES = 33 #background not included
N_NAMES = 5
N_SHAPES = 3
N_COLORS = 4

global_config = {
    'model_dir': os.path.dirname(os.path.abspath(__file__)) + '/model/',
    'progress_dir': os.path.dirname(os.path.abspath(__file__)) + '/progress/',
    'data_dir': os.path.dirname(os.path.abspath(__file__)) + '/datasets/RD_dataset/',
}
sys.path.extend([global_config['model_dir'], os.path.dirname(os.path.dirname(os.path.abspath(__file__)))])

LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_name_loss',
                        'roi_color_loss',
                        'total_loss'
                        ])


class Trainer:
    def __init__(self, args):
        self.conf = args
        self.check_cuda()
        if not os.path.isdir(global_config['progress_dir'] + self.conf.model):
            os.mkdir(global_config['progress_dir'] + self.conf.model)
        self.tp = TrainingProgress(global_config['progress_dir'] + self.conf.model + '/',
                                   self.conf.net_model + '-progress',
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
        if_pretrained = not self.conf.res  # if not loading weight
        if self.conf.net_model in ['VGG_16', 'VGG_16_bn', 'VGG_19', 'VGG_19_bn']:
            self.net = FasterRCNN_VGG(version=self.conf.net_model,
                                      n_names=N_NAMES, n_colors=N_COLORS,
                                      freeze_idx=self.conf.freeze_idx,
                                      vgg_pre=if_pretrained, use_drop= self.conf.use_drop)  # contains rpn and head(classification)
        elif self.conf.net_model in ['ResNet_18','ResNet_34','ResNet_50','ResNet_101', 'ResNet_152']: #if use resnet
            self.net = FasterRCNN_ResNet(version=self.conf.net_model,
                                         n_names=N_NAMES, n_colors=N_COLORS,
                                         freeze_idx=self.conf.freeze_idx,
                                         pretrained=if_pretrained, use_drop= self.conf.use_drop) #contains rpn and head(classification)
        else:
            raise ValueError('The net model not exist')
        self.net = self.net.to(self.device)
        # Load Target Generators for Training
        self.anchor_target = AnchorTargetCreator()
        self.proposal_target = ProposalTargetCreator()
        print(self.conf.net_model + ' is used')
    def load_data(self,args):  # batch_files, data_conf, cuda_dev):
        """
        Load Training/Testing data
        """

        pp_train = Preprocess(normalize_means=[0.485, 0.456, 0.406], normalize_stds=[0.229, 0.224, 0.225], p_hflip=0.5)
        pp_test = Preprocess(normalize_means=[0.485, 0.456, 0.406], normalize_stds=[0.229, 0.224, 0.225], p_hflip=0,
                             bbox_resize=False)
        self.train_set = RD_dataset(data_dir=global_config['data_dir'], split='trainval',
                                       preprocess=pp_train)
        self.test_set = RD_dataset(data_dir=global_config['data_dir'], split='test',
                                      preprocess=pp_test)
        print('Data loader: Training set: ', len(self.train_set), ' Testing set: ', len(self.test_set))

    def create_data_loader(self):

        self.train_loader = DataLoader(dataset=self.train_set, batch_size=self.conf.tr_bat, shuffle=True,
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
                for step, (img_name, img, index, bbox, name, shape, color, _, scale) in tqdm(enumerate(self.train_loader)): #bbox shapeis [1, 2, 4] #label is [1,n] label_attr is[1,m,n]
                    #a = input("label is {2}, label_shape is {0}, label_color is {1}".format(label_shape.numpy().shape, label_color.numpy().shape, label.numpy().shape))
                    self.optimizer.zero_grad()
                    img, bbox, label, scale = img.to(self.conf.dev), bbox.to(self.conf.dev), name.to(self.conf.dev), at.scalar(scale)
                    loss = self.train_forward_net(img_name, img, index, bbox, name, shape, color, _, scale)
                    loss.total_loss.backward()
                    self.optimizer.step()
                    self.update_meters(loss)
                    self.epoch_loss += loss.total_loss.detach().cpu().numpy() * img.size(0)


                    #if step == 5: break #for test

                    # if step % 200 == 0:
                    #     print('Step=', step)
                # ['epoch_loss', 'test_loss', 'training_loss']
                if self.epoch == 1: ##create record file
                    with open(global_config['progress_dir'] + self.conf.net_model + '.txt', 'w') as f:
                        f.write('lr_rates: {0}, lr interval: {1}, use_drop: {2}\n'
                                'freeze_idx: {3}, opt: {4}, w_decay: {5}\n'\
                                .format(self.conf.lr_rates,self.conf.lr_epochs, self.conf.use_drop,
                                        self.conf.freeze_idx, self.conf.optim, self.conf.w_decay))
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

    def train_forward_net(self, img_names, imgs, indexs, bboxes, names, shapes, colors, _, scale):

        n = bboxes.shape[0]
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = imgs.shape
        img_size = (H, W)
        #print("img_name {0}".format(img_name))
        #vis_image(imgs.cpu()[0])
        #plt.show()
        features = self.net.extractor(imgs) ##vgg16 (1,512,37,50)
        #print(features.shape)
        rpn_locs, rpn_scores, rois, _, anchor = self.net.rpn(features, img_size, scale)
        # Since batch size is one, convert variables to singular form
        bbox = bboxes[0]
        name = names[0]
        color = colors[0]

        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois

        # Sample RoIs(For the training of head(classification network)) and forward

        sample_roi, gt_roi_loc, gt_roi_name, gt_roi_color = self.proposal_target(
            roi,
            at.to_np(bbox),
            at.to_np(name),
            at.to_np(color)
        )
        # NOTE it's all zero because now it only support for batch=1 now
        sample_roi_index = torch.zeros(len(sample_roi))
        roi_name_loc, roi_name_score = self.net.head_name(
            features,
            sample_roi,
            sample_roi_index)

        _, roi_color_score = self.net.head_color(
            features,
            sample_roi,
            sample_roi_index)

        # ------------------ RPN losses -------------------#
        # Target for RPN => Anchor Target
        gt_rpn_loc, gt_rpn_label = self.anchor_target(
            at.to_np(bbox),
            anchor,
            img_size)

        gt_rpn_label = at.to_tensor(gt_rpn_label).long()
        gt_rpn_loc = at.to_tensor(gt_rpn_loc)
        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_loc,
            gt_rpn_loc,
            gt_rpn_label.data,
            self.conf.rpn_sigma)

        # NOTE: default value of ignore_index is -100 ...
        rpn_cls_loss = torch.nn.functional.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)
        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        _rpn_score = at.to_np(rpn_score)[at.to_np(gt_rpn_label) > -1]
        self.rpn_cm.add(at.to_tensor(_rpn_score, False), _gt_rpn_label.data.long())

        # ------------------ ROI losses (fast rcnn loss) -------------------#
        ##name##
        n_sample = roi_name_loc.shape[0]
        roi_name_loc = roi_name_loc.view(n_sample, -1, 4)
        name_roi_loc = roi_name_loc[torch.arange(0, n_sample).long().cuda(), \
                              at.to_tensor(gt_roi_name).long()] ## not one-hot
        gt_roi_name = at.to_tensor(gt_roi_name).long()
        gt_roi_loc = at.to_tensor(gt_roi_loc)

        roi_loc_loss = _fast_rcnn_loc_loss(
            name_roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_name.data,
            self.conf.roi_sigma)
        roi_name_loss = nn.CrossEntropyLoss()(roi_name_score, gt_roi_name.cuda())

        self.roi_name_cm.add(at.to_tensor(roi_name_score, False), gt_roi_name.data.long())

        ##color##
        gt_roi_color = at.to_tensor(gt_roi_color).long()
        roi_color_loss = nn.CrossEntropyLoss()(roi_color_score, gt_roi_color.cuda())

        self.roi_color_cm.add(at.to_tensor(roi_color_score, False), gt_roi_color.data.long())


        ##sum up all loss##
        losses = [rpn_loc_loss, rpn_cls_loss,
                  roi_loc_loss, roi_name_loss, roi_color_loss]
        losses = losses + [sum(losses)]

        return LossTuple(*losses)

    # Meters
    def init_meters(self):
        self.rpn_cm = ConfusionMeter(2) #forground and background
        self.roi_name_cm = ConfusionMeter(N_NAMES+1) #num of class(including background)
        self.roi_color_cm = ConfusionMeter(N_COLORS + 1)  # num of class(including background)
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}  # average loss

    def update_meters(self, losses):
        loss_d = {k: at.scalar(v) for k, v in losses._asdict().items()}
        for key, meter in self.meters.items():
            meter.add(loss_d[key])

    def reset_meters(self):
        for key, meter in self.meters.items():
            meter.reset()
        self.rpn_cm.reset()
        self.roi_name_cm.reset()
        self.roi_color_cm.reset()

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

        test_map = self.eval_faster_rcnn()
        # valid_loss = self.test(use_validation=True) #Already record in training loop

        self.tp.record_data({'mAP_name': test_map['map_name'], 'mAP_color': test_map['map_color']}, display=True)
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
        torch.save(self.net.state_dict(),
                   '{0}{1}_{2}.pkl'.format(global_config['progress_dir'], self.conf.net_model, self.epoch))
        if display:
            print('Progress Saved, current epoch=', self.epoch)
        self.net.state_dict()
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

    def eval_faster_rcnn(self, visualize = False):
        # list below store all img bboxes, the first dim is img index, second is which bbox, third is the label or coordinate
        pred_bboxes, pred_names, pred_name_scores, pred_colors, pred_color_scores  = [], [], [], [], []
        gt_bboxes, gt_names, gt_colors = [], [], []
        img_count = 0
        imgs = []
        for ii, (img_names, img, indexs, gt_bboxes_, gt_names_, gt_shapes_, gt_colors_, _, scale) \
                in tqdm(enumerate(self.test_loader), total=len(self.test_loader)): #gt_label is numpy (1,N物體)

            pred_bboxes_, pred_names_, pred_name_scores_, pred_colors_, pred_color_scores_ = \
                self.net.predict(img, scale, visualize=True)
            #print("predict shape {0}".format(pred_bboxes_.shape[0]))
            gt_bboxes.append(gt_bboxes_[0].numpy())
            gt_names.append(gt_names_[0].numpy())
            gt_colors.append(gt_colors_[0].numpy())
            """
            print("shape of predictions")
            print("bbox{0}".format(pred_bboxes_.shape))
            print("label{0}".format(pred_labels_.shape))
            print("label_score{0}".format(pred_scores_.shape))
            print("shape{0}".format(pred_labels_shape_.shape))
            print("shape_score{0}".format(pred_scores_shape_.shape))
            print("color{0}".format(pred_labels_color_.shape))
            print("color_score{0}".format(pred_scores_color_.shape))
            """

            pred_bboxes.append(pred_bboxes_)
            pred_names.append(pred_names_)
            pred_name_scores.append(pred_name_scores_)
            pred_colors.append(pred_colors_)
            pred_color_scores.append(pred_color_scores_)
            img_count += 1
            imgs.append(img)
            ####visualize the label####
            if visualize and pred_names[-1].shape[0] > 0:
                gt_arg_list = [gt_bboxes[-1], gt_names[-1], gt_colors[-1]]
                pred_arg_list = [pred_bboxes[-1], pred_names[-1], pred_name_scores[-1], pred_colors[-1], pred_color_scores[-1]]
                vis_bbox(img[0], *gt_arg_list, *pred_arg_list)
                plt.show()
            ###########################
            #if ii ==5: break ##for test
        ##result is a dict
        #print(len(pred_bboxes), len(gt_bboxes))
        result = eval_detection_voc(
            pred_bboxes, pred_names, pred_name_scores, pred_colors, pred_color_scores,
            gt_bboxes, gt_names, gt_colors,
            use_07_metric=False,
            show_prec=self.conf.show_prec, show_rec=self.conf.show_rec
        ) #not one-hot type

        print('map_name: {0}, map_color: {1}'.format(result['map_name'], result['map_color']))

        ##write info to file
        with open(global_config['progress_dir'] + self.conf.net_model + '.txt', 'a') as f:
            f.write('epoch= {0}, epoch_loss= {1}\n'.format(self.epoch, self.epoch_loss))
            f.write('map_name: {0}, map_color: {1}\n'.format(result['map_name'], result['map_color']))
        return result


def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    flag = at.to_tensor(flag)
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = torch.zeros(gt_loc.shape).cuda()
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation,
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, at.to_tensor(in_weight), sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= (gt_label >= 0).sum().to(torch.float)  # ignore gt_label==-1 for rpn_loss
    return loc_loss


def main():
    global cf, Net
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', help='Model file name', type=str, default='obj_detect_model_shuffle')
    parser.add_argument('-net_model', help='net Model(vgg or resnet)', type=str, default='ResNet_101') ##new
    parser.add_argument('-freeze_idx', help='all block weight before(include itself) this index will be frozen',
                        type= int, default=-1)
    parser.add_argument('--use_drop', help='whether use dropout', action='store_true', default=False)
    parser.add_argument('-opt_header', help='Additional weight/progress file header', type=str, default='')
    parser.add_argument('-dev', help='CUDA Device Number, used when GPU is available', type=int, default=1)
    parser.add_argument('-tr', help='Training Data Ratio (Validation ration is 1-tr)', type=float, default=0.95)

    # Learning rate scheduler setting
    parser.add_argument('-lrs', help='Learning rate scheduler mode', type=str, default='epoch')
    parser.add_argument('-lr_init', help='Initial Learning rate for Torch lr Schedulers', type=float, default=1e-3)
    parser.add_argument('-lr_rates', help='Learning Rates for epochs.', nargs='+', type=float,
                        default=[1e-3, 1e-4, 5e-5])
    parser.add_argument('-lr_epochs', help='Epochs for Learning Rate control.', nargs='+', type=int, default=[7, 10])
    parser.add_argument('-lr_loss', help='Loss targets for Learning Rate control.', nargs='+', type=float, default=None)
    parser.add_argument('--show_rec', help='Whether show recall rate each epoch', action='store_true', default=False)
    parser.add_argument('--show_prec', help='Whether show precision rate each epoch', action='store_true', default=False)
    parser.add_argument('-optim', help='Overwrite the optimizer, default=SGD.',
                        choices=['RMSprop', 'SGD', 'Adadelta', 'Adam'], default='SGD')
    parser.add_argument('-w_decay', help='Weight decay parameter for Optimizer. Ex: 1e-5', type=float, default=1e-5)

    parser.add_argument('-se', help='Save progress and do summary every ? epoch', type=int, default=1)
    parser.add_argument('--res', help='Restore progress and resume the training progress',
                        action='store_true', default=False)
    parser.add_argument('-tps', help='Restore Training Progress index(step)', type=int, default=15)
    parser.add_argument('--save_opt', help='Save optimizer State Dict (Take some time !)?', action='store_true',
                        default=False)

    parser.add_argument('--test', help='Display Testing Result only!', action='store_true', default=False)

    parser.add_argument('-tr_bat', help='Training Batch Size', type=int, default=1)
    parser.add_argument('-ts_bat', help='Testing Batch Size', type=int, default=1)

    parser.add_argument('-max_epoch', help='Max Epoch for training', type=int, default=20) ##15 orginal
    parser.add_argument('-min_loss', help='Minimum Loss for training', type=float, default=1e-5)
    parser.add_argument('--init_fin', help='Weight Initialization by the fan-in size', action='store_true',
                        default=False)
    parser.add_argument('--para', help='Parallel Training', action='store_true', default=False)
    """
    Data Preprocessing and Loss Setting
    """
    parser.add_argument('-rpn_sigma', type=float, default=3.0)
    parser.add_argument('-roi_sigma', type=float, default=1.0)

    # parser.add_argument('-test_epoch', help='Do test every ? epoch', type=int, default=10)
    args = parser.parse_args()
    trainer = Trainer(args)
    if args.test:
        assert args.res
        trainer.eval_faster_rcnn(visualize=False)
    else:
        trainer.train()
        # trainer.eval_faster_rcnn()


if __name__ == '__main__':
       main()
