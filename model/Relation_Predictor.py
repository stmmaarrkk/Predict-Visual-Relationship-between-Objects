from model.faster_rcnn_vgg16 import decom_vgg16
import torch
import torch.nn as nn
import numpy as np
from .roi_module import RoIPooling2D
from .utils.combine_feature import get_combined_feature
from ipdb import set_trace
cfg = { ##dim 1 is out channel, dim 2 is kernel size
    'A':[[256, 1], [256,3], [512, 1], [64, 3]],
    'B':[[32, 3], [64, 3], [128, 3], [64, 3]]
}
def make_layers(cfg_set, batch_norm=False):
    layers = []
    in_channels = 0
    if cfg_set == 'A':
        in_channels = 512
    elif cfg_set == 'B':
        in_channels = 2
    used_cfg = cfg[cfg_set]
    for v in used_cfg:
        if v[0] == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v[0], kernel_size=v[1], padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v[0]), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v[0]
    return nn.Sequential(*layers)

class Conv_unit(nn.Module):
    def __init__(self, cfg_set='A'):
        super(Conv_unit, self).__init__()
        self.seq = make_layers(cfg_set, True)
    def forward(self, input):
        output = self.seq(input)
        return output

class Relationship_Predictor(nn.Module):
    def __init__(self, n_rel_class=3, spatial_feature=False, use_extractor=False, pretrained=False):
        super(Relationship_Predictor, self).__init__()
        if use_extractor:
            self.extractor, _ = decom_vgg16(pretrained)
        self.use_spatial = spatial_feature
        ##conv part
        self.obj_a_conv = Conv_unit()
        self.obj_b_conv = Conv_unit()
        self.rel_conv = Conv_unit()

        self.fc1 = nn.Linear(in_features=23232, out_features=4096)  # 23232 may be changed
        if self.use_spatial:
            self.spatial_conv = Conv_unit(cfg_set='B')
            self.fc1 = nn.Linear(in_features=23232+118400, out_features=4096)  # 9408 may be changed
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=4096, out_features=1024)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc_part = nn.Sequential(self.fc1, self.relu1, self.fc2, self.relu2)
        ##score part
        self.relation_score = nn.Linear(in_features=1024, out_features=n_rel_class)
        normal_init(self.relation_score, 0, 0.01)
    def forward(self, *input):
        obj_a, obj_b, rel = input[0:3]
        sp = None
        if self.use_spatial:
            assert len(input) == 4
            sp = input[3]
        output_a = self.obj_a_conv(obj_a) #[1,64,11,11]
        output_b = self.obj_b_conv(obj_b)
        output_rel = self.rel_conv(rel)
        output = torch.stack((output_a, output_b, output_rel), dim=0)
        output = output.view(1, -1) #[1, 3*64*11*11]
        if self.use_spatial:
            output_sp = self.spatial_conv(sp)
            output_sp = output_sp.view(1,-1)
            output = torch.cat((output, output_sp), dim=0)
        output = self.fc_part(output)
        rel_score = self.relation_score(output) #[1, n_class]
        return rel_score
    def predict(self, img, bbox): #need to be in numpy with shape[N] and not one-hot
        """
        :param img: tensor type of shape [1,3,600,800]
        :param bbox: tensor type of shape [N, 4]
        :param scale:
        :return:
        """
        self.eval()
        feature = self.extractor(img)

        combined_features, _ = get_combined_feature(feature, bbox, flip=False, use_spatial_feature=self.use_spatial) #don't flip the relation
        #set_trace()
        relation_scores = []
        for combined_feature in combined_features:
            assert len(combined_feature) > 0
            relation_score = self(*combined_feature)
            relation_scores.append(relation_score)
        #print(len(relation_scores))
        relation_scores = torch.stack(relation_scores, dim=1).squeeze(0) #[N, n_class]
        relation_scores = relation_scores.cpu().detach().numpy()

        relations = np.empty(shape=relation_scores.shape[0], dtype=int)
        for i in range(relation_scores.shape[0]):
            relations[i] = np.argmax(relation_scores[i])
        self.train()
        return  relations, relation_scores

def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()