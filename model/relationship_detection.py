from model.faster_rcnn_resnet import decom_resnet
import torch
import torch.nn as nn

class Relationship_Detection(nn.Module):
    def __init__(self, n_obj_class=3, n_rel_class=5, pretrained=False):
        super(Relationship_Detection, self).__init__()
        appr_front, _ = decom_resnet(pretrained) ##the front side
        self.appearance = Appearance(n_out=256, appr_front=appr_front) #combine to whole appr network
        self.spatial = Spatial(n_out=64)
        self.combine = Combine(n_in=2048, n_obj_class=n_obj_class, n_rel_class=n_rel_class)
    def forward(self, dual_channel_img, cropped_img):
        spatial_feature = self.spatial(dual_channel_img)
        appearance_feature = self.appearance(cropped_img)
        subject_score, relation_score, object_score = self.combine(spatial=spatial_feature, appr=appearance_feature)
        return subject_score, relation_score, object_score
    def predict(self, dual_channel_img, cropped_img):
        self.eval()
        subject_score, relation_score, object_score = self(dual_channel_img, cropped_img) #go to forward
        self.train()
        return subject_score, relation_score, object_score

class Spatial(nn.Module):
    def __init__(self, n_out):
        super(Spatial, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=96, kernel_size=5, stride=2, padding=2)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=n_out, kernel_size=8, stride=2, padding=2)
        self.relu3 = nn.ReLU(inplace=False)
        self.seq = nn.Sequential(self.conv1, self.relu1, self.conv2, self.conv3, self.relu3)
    def forward(self, input):
        output = self.seq(input)
        return output

class Appearance(nn.Module):
    def __init__(self, n_out, appr_front):
        super(Appearance, self).__init__()
        self.front = appr_front
        self.fc7 = nn.Linear(in_features=2048, out_features=4096)
        self.relu1 = nn.ReLU(inplace=False)
        self.fc8 = nn.Linear(in_features=4096, out_features=n_out)
        self.relu2 = nn.ReLU(inplace=False)
        self.seq = nn.Sequential(self.front, self.fc7, self.relu1, self.fc8, self.relu2)
    def forward(self, input):
        output = self.seq(input)
        return output
"""
class Joint(nn.Module):
    def __init__(self):
        super(Joint, self).__init__()
"""
class Combine(nn.Module):
    def __init__(self, n_in, n_obj_class, n_rel_class):
        super(Combine, self).__init__()
        self.fc1 = nn.Linear(in_features=n_in, out_features=128)
        self.relu1 = nn.ReLU(inplace=False)
        self.fc2 = nn.Linear(in_features=128, out_features=70)
        self.relu2 = nn.ReLU(inplace=False)
        self.subject_score = nn.Linear(in_features=70, out_features=3)
        self.relation_score = nn.Linear(in_features=70, out_features=5)
        self.object_score = nn.Linear(in_features=70, out_features=3)
        normal_init(self.subject_score, 0, 0.01)
        normal_init(self.relation_score, 0, 0.01)
        normal_init(self.object_score, 0, 0.01)
        self.seq = nn.Sequential(self.fc1, self.relu1, self.fc2, self.relu2)
    def forward(self, spatial, appr):
        x = spatial.view(1,-1)
        y = appr.view(1,-1)
        input = torch.cat([x[1,:],y[1,:]], dim=1)
        assert input.size(1) == x.size(1)+y.size(1)
        output = self.seq(input)
        subject_score = self.subject_score(output)
        relation_score = self.relation_score(output)
        object_score = self.object_score(output)
        return subject_score, relation_score, object_score
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