import numpy as np
import time, math
import torch
from torch.utils.data import Dataset
from torch import nn
from PIL import Image
import pickle
from torch.optim import lr_scheduler
import utils.array_tool as at


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def secondSince(since):
    now = time.time()
    s = now - since
    return s


class TrainingProgress:
    def __init__(self, path, header, tp_step=None, data_key_list=None, data_dict=None, meta_dict=None, restore=False):
        """
        * Init Dict for storing key-value data
        * Configure Saving filename and Path
        * Restoring function

        Header => Filename header
        Data Dict => Appendable data (loss,time,acc....)
        Meta Dict => One time data (config,weight,.....)
        """
        self.filename = path + header
        if restore:
            assert tp_step is not None, 'Explicitly assign the TP step you want to restore'
            self.restore_progress(tp_step)
        else:  # Two Initialization Methods for data_dict, keys or a dict, select one
            if data_key_list is not None:
                self.data_dict = {}
                for k in data_key_list:
                    self.data_dict[k] = []
            else:
                self.data_dict = {} if data_dict is None else data_dict
            self.meta_dict = {} if meta_dict is None else meta_dict

    def add_meta(self, new_dict):
        self.meta_dict.update(new_dict)

    def get_meta(self, key):
        try:
            return self.meta_dict[key]
        except KeyError:  # New key
            print('TP Error: Cannot find meta, key=', key)
            return None

    def record_data(self, new_dict, display=False):
        for k, v in new_dict.items():
            try:
                # if math.isnan(v):
                #     print('TP Warning: Ignore NaN value')
                # else:
                self.data_dict[k].append(v)
            except AttributeError:  # Append fail
                print('TP Error: Cannot Record data, key=', k)
            except KeyError:  # New key
                print('TP Warning: Add New Appendable data, key=', k)
                self.data_dict[k] = [v]
        if display:
            print('TP Record new data: ', new_dict)
        pass

    def save_progress(self, tp_step):
        with open(self.filename + str(tp_step) + '.tpdata', "wb") as f:
            pickle.dump((self.data_dict, self.meta_dict), f, protocol=2)

    def restore_progress(self, tp_step):
        with open(self.filename + str(tp_step) + '.tpdata', 'rb') as f:
            self.data_dict, self.meta_dict = pickle.load(f)


class LearningRateScheduler:  # Include torch.optim.lr_scheduler
    def __init__(self, mode, optimizer, lr_rates, lr_epochs, lr_loss, lr_init, lr_decay_func,
                 torch_lrs='ReduceLROnPlateau', torch_lrs_param={'mode': 'min', 'factor': 0.5, 'patience': 20}):
        self.mode = mode
        self.optimizer = optimizer
        self.rate = lr_init
        # Check each mode
        if self.mode == 'epoch':
            self.lr_rates = lr_rates  # only single value if decay mode else list of rate
            self.epoch_targets = lr_epochs
            assert (0 <= len(self.lr_rates) - len(self.epoch_targets) <= 1), "Learning rate scheduler setting error."
            self.rate_func = self.lr_rate_epoch
            self.adjust_learning_rate(self.rate)

        elif self.mode == 'loss':
            self.lr_rates = lr_rates  # only single value if decay mode else list of rate
            self.loss_targets = lr_loss
            assert (0 <= len(self.lr_rates) - len(self.loss_targets) <= 1), 'Learning rate scheduler setting error.'
            self.rate_func = self.lr_rate_loss
            self.adjust_learning_rate(self.rate)

        elif self.mode == 'decay':
            self.decay_func = lr_decay_func
            self.rate_func = self.lr_rate_decay
            raise NotImplementedError  # Zzz....

        elif self.mode == 'torch':
            # Should set the lr scheduler name in torch.optim.scheduler
            assert torch_lrs_param is not None, "Learning rate scheduler setting error."

            if torch_lrs == 'ReduceLROnPlateau':
                self.torch_lrs = getattr(lr_scheduler, 'ReduceLROnPlateau')(self.optimizer,
                                                                            **torch_lrs_param)  # instance
            else:
                raise NotImplementedError
            self.rate_func = self.torch_lrs.step
        else:
            raise NotImplementedError("Learning rate scheduler setting error.")
        print('Learning rate scheduler: Mode=', self.mode, ' Learning rate=', self.rate)

    def step(self, param_dict, display=True):
        if self.mode == 'torch':
            self.rate_func(param_dict[self.mode])
        else:
            new_rate, self.next = self.rate_func(param_dict[self.mode])
            if new_rate == self.rate:
                return
            else:
                self.rate = new_rate
                if display:
                    print('Learning rate scheduler: Mode=', self.mode, ' New Learning rate=', new_rate,
                          ' Next ', self.mode, ' target=', self.next)
                self.adjust_learning_rate(self.rate)

    def lr_rate_epoch(self, epoch):
        for idx, e in enumerate(self.epoch_targets):
            if epoch < e:
                # next lr rate, next epoch torget for changing lr rate
                return self.lr_rates[idx], self.epoch_targets[idx]
        return self.lr_rates[-1], -1  # Last(smallest) lr rate

    def lr_rate_loss(self, loss):
        for idx, l in enumerate(self.loss_targets):
            if loss > l:
                return self.lr_rates[idx], self.loss_targets[idx]  # next lr rate, next loss target for changing lr rate
        return self.lr_rates[-1], -1  # Last(smallest) lr rate

    def lr_rate_decay(self, n):
        rate = self.lr_rates * self.decay_func(n)
        return rate, -1

    def adjust_learning_rate(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


def initialize_weight(net, bias=1e-4):
    for name, param in net.state_dict().items():
        if name.find('weight') != -1:
            if len(param.size()) >= 2:
                print('Init Weight:', name, ' size=', param.size())
                nn.init.kaiming_normal_(param)
        elif name.find('bias') != -1:
            nn.init.constant_(param, bias)
    print('All Weight Initialized')


def initialize_weight2(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


# def initialize_weight_fin(net, mean=0, bias=1e-4):
#     for name, param in net.state_dict().items():
#         if name.find('weight') != -1:
#             size = param.size()  # returns a tuple
#             print('Init weight name',name,' size:', size)
#             # fan_out = size[0]  # number of rows
#             fan_in = size[1]  # number o
#             nn.init.normal(param, mean=mean, std=np.sqrt(1 / fan_in))
#         elif name.find('bias') != -1:
#             nn.init.constant(param, bias)
#     print('All Weight Initialized (Fan-in mode)')


class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def train_valid_split(dataset, train_ratio, random_indices=None):
    N = len(dataset)
    train_n = int(train_ratio * N)
    valid_n = N - train_n
    assert train_ratio <= 1
    print('Training set:', train_n, ' , Validation set:', valid_n)
    indices = random_indices if random_indices is not None else np.random.permutation(N)
    assert len(indices) == N
    return Subset(dataset, indices=indices[0:train_n]), Subset(dataset, indices=indices[train_n:N])


class SimpleNetLoader:
    def __init__(self, header, tp_step, load_net=True):
        from trainer import global_config
        self.check_cuda()
        self.tp = TrainingProgress(global_config['progress_dir'], 'model_A1' + header + '-progress', tp_step=tp_step,
                                   restore=True)
        if load_net:
            self.load_net()

    def load_net(self):
        from model.faster_rcnn_vgg16 import FasterRCNNVGG16
        self.net = FasterRCNNVGG16()
        self.net = self.net.to('cpu')
        self.net.load_state_dict(self.tp.get_meta('net_weight'))
        self.net = self.net.to(self.device)

    def check_cuda(self, dev=None):
        if dev is not None:
            self.device = dev
        else:
            if torch.cuda.is_available():
                torch.cuda.set_device(0)
                self.device = 'cuda:' + str(0)
                print("Use CUDA,device=", torch.cuda.current_device())
                at.gpu_dev = self.device
            else:
                self.device = 'cpu'
