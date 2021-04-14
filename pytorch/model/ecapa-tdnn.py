''' Implementation of
    "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification".

    Note that we DON'T concatenate the last frame-wise layer with non-weighted mean and standard deviation, 
    because it brings little improvment but significantly increases model parameters. 
    As a result, this implementation basically equals the A.2 of Table 2 in the paper.
'''

import os
import sys
import math
import torch
import torch.nn.functional as F
import torch.nn as nn

subtools = '/data/lijianchen/workspace/sre/subtools'
# subtools = os.getenv('SUBTOOLS')
sys.path.insert(0, '{}/pytorch'.format(subtools))

import libs.support.utils as utils
from libs.nnet import *


class Res2Conv1dReluBn(nn.Module):
    '''
    Res2Conv1d + BatchNorm1d + ReLU

    in_channels == out_channels == channels
    '''
    def __init__(self, channels, context=[0], bias=False, scale=4, inplace=True,
                 affine_type="tdnn-affine", bn_params={}):
        super().__init__()
        default_bn_params = {"momentum": 0.1, "affine": True, "track_running_stats": True}
        bn_params = utils.assign_params_dict(default_bn_params, bn_params)

        assert channels % scale == 0, "{} % {} != 0".format(channels, scale)
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        self.convs = []
        self.bns = []
        for i in range(self.nums):
            self.convs.append(ReluBatchNormTdnnLayer(self.width, self.width, context, affine_type,
                                                     bias=bias, nonlinearity="", bn=False))
            self.bns.append(nn.BatchNorm1d(self.width, **bn_params))
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            # Order: conv -> relu -> bn
            sp = self.convs[i](sp)
            sp = self.bns[i](self.relu(sp))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        out = torch.cat(out, dim=1)
        return out


def SE_Res2Block(channels, context, scale, tdnn_layer_params={}):
    ''' SE-Res2Block.
        Note: residual connection is implemented in the ECAPA_TDNN model, not here.
    '''
    return nn.Sequential(
        ReluBatchNormTdnnLayer(channels, channels, **tdnn_layer_params),
        Res2Conv1dReluBn(channels, context, scale=scale, bn_params=tdnn_layer_params["bn_params"],
                        affine_type=tdnn_layer_params["affine_type"]),
        ReluBatchNormTdnnLayer(channels, channels, **tdnn_layer_params),
        SEBlock(channels, ratio=4, affine_type=tdnn_layer_params["affine_type"])
    )


class AttentiveStatsPool(nn.Module):
    ''' Attentive weighted mean and standard deviation pooling.
    '''
    def __init__(self, in_dim, bottleneck_dim, affine_type):
        super().__init__()
        # Use Conv1d with stride == 1 rather than Linear, then we don't need to transpose inputs.
        self.linear1 = ReluBatchNormTdnnLayer(in_dim, bottleneck_dim, affine_type=affine_type, nonlinearity="", bn=False)
        self.linear2 = ReluBatchNormTdnnLayer(bottleneck_dim, in_dim, affine_type=affine_type, nonlinearity="", bn=False)

    def forward(self, x):
        # DON'T use ReLU here! In experiments, I find ReLU hard to converge.
        alpha = torch.tanh(self.linear1(x))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2, keepdim=True)
        residuals = torch.sum(alpha * x ** 2, dim=2, keepdim=True) - mean ** 2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)


class ECAPA_TDNN(TopVirtualNnet):
    def init(self, inputs_dim, num_targets, channels=512, emb_dim=192,
             tdnn_layer_params={}, layer5_params={}, layer6=False, layer7_params={},
             margin_loss=False, margin_loss_params={}, pooling="statistics",
             use_step=False, step_params={}, training=True, extracted_embedding="near"):
        default_tdnn_layer_params = {
            "affine_type": 'tdnn-affine',
            "nonlinearity": 'relu', "nonlinearity_params": {"inplace": True},
            "bn-relu": False, "bn": True, "bn_params": {"momentum": 0.5, "affine": False, "track_running_stats": True}
        }
        default_layer5_params = {"nonlinearity": 'relu', "bn": False}
        default_layer7_params = {"nonlinearity": '', "bn": True}
        default_margin_loss_params = {
            "method": "am", "m": 0.2,
            "feature_normalize": True, "s": 30,
            "double": False,
            "mhe_loss": False, "mhe_w": 0.01,
            "inter_loss": 0.,
            "ring_loss": 0.,
            "curricular": False
        }

        default_step_params = {
            "T": None,
            "m": False, "lambda_0": 0, "lambda_b": 1000, "alpha": 5, "gamma": 1e-4,
            "s": False, "s_tuple": (30, 12), "s_list": None,
            "t": False, "t_tuple": (0.5, 1.2),
            "p": False, "p_tuple": (0.5, 0.1)
        }

        tdnn_layer_params = utils.assign_params_dict(default_tdnn_layer_params, tdnn_layer_params)
        layer5_params = utils.assign_params_dict(default_layer5_params, layer5_params)
        layer5_params = utils.assign_params_dict(default_tdnn_layer_params, layer5_params)
        layer7_params = utils.assign_params_dict(default_layer7_params, layer7_params)
        layer7_params = utils.assign_params_dict(default_tdnn_layer_params, layer7_params)
        margin_loss_params = utils.assign_params_dict(default_margin_loss_params, margin_loss_params)
        step_params = utils.assign_params_dict(default_step_params, step_params)

        self.use_step = use_step
        self.step_params = step_params
        self.extracted_embedding = extracted_embedding  # For extract.

        self.layer1 = ReluBatchNormTdnnLayer(inputs_dim, channels, [-2, -1, 0, 1, 2], **tdnn_layer_params)
        # channels, kernel_size, stride, padding, dilation, scale
        self.layer2 = SE_Res2Block(channels, [-2, 0, 2], 8, tdnn_layer_params)
        self.layer3 = SE_Res2Block(channels, [-3, 0, 3], 8, tdnn_layer_params)
        self.layer4 = SE_Res2Block(channels, [-4, 0, 4], 8, tdnn_layer_params)

        cat_channels = channels * 3
        self.layer5 = ReluBatchNormTdnnLayer(cat_channels, cat_channels, **layer5_params)

        if pooling == "attention":
            self.pooling = AttentiveStatsPool(cat_channels, 128, tdnn_layer_params["affine_type"])
        else:
            self.pooling = StatisticsPooling(cat_channels, stddev=True)

        # self.bn1 = nn.BatchNorm1d(cat_channels * 2, **tdnn_layer_params["bn_params"])


        # Segment level
        if layer6:
            self.layer6 = ReluBatchNormTdnnLayer(cat_channels * 2, 512, **tdnn_layer_params)
            layer7_dim = 512
        else:
            self.layer6 = None
            layer7_dim = cat_channels * 2
        self.layer7 = ReluBatchNormTdnnLayer(layer7_dim, emb_dim, **layer7_params)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')

        if training:
            if margin_loss:
                self.loss = MarginSoftmaxLoss(emb_dim, num_targets, **margin_loss_params)
            else:
                self.loss = SoftmaxLoss(emb_dim, num_targets, affine_type=tdnn_layer_params["affine_type"])

    @utils.for_device_free
    def forward(self, inputs):
        """
        inputs: [batch, features-dim, frames-lens]
        """
        out1 = self.layer1(inputs)
        out2 = self.layer2(out1) + out1
        out3 = self.layer3(out1 + out2) + out1 + out2
        out4 = self.layer4(out1 + out2 + out3) + out1 + out2 + out3

        # 在 channel 维连接
        out = torch.cat([out2, out3, out4], dim=1)
        out = self.layer5(out)
        # out = self.bn1(self.pooling(out))
        out = self.pooling(out)
        out = self.auto(self.layer6, out)
        out = self.layer7(out)
        return out
      
    @utils.for_device_free
    def get_loss(self, inputs, targets):
        """Should call get_loss() after forward() with using Xvector model function.
        e.g.:
            m=Xvector(20,10)
            loss=m.get_loss(m(inputs),targets)

        model.get_loss [custom] -> loss.forward [custom]
          |
          v
        model.get_accuracy [custom] -> loss.get_accuracy [custom] -> loss.compute_accuracy [static] -> loss.predict [static]
        """
        return self.loss(inputs, targets)

    @utils.for_device_free
    def get_accuracy(self, targets):
        """Should call get_accuracy() after get_loss().
        @return: return accuracy
        """
        return self.loss.get_accuracy(targets)

    @for_extract_embedding(maxChunk=10000, isMatrix=True)
    def extract_embedding(self, inputs):
        """
        inputs: a 3-dimensional tensor with batch-dim = 1 or normal features matrix
        return: an 1-dimensional vector after processed by decorator
        """
        out1 = self.layer1(inputs)
        out2 = self.layer2(out1) + out1
        out3 = self.layer3(out1 + out2) + out1 + out2
        out4 = self.layer4(out1 + out2 + out3) + out1 + out2 + out3

        # 在 channel 维连接
        out = torch.cat([out2, out3, out4], dim=1)
        out = self.layer5(out)
        # out = self.bn1(self.pooling(out))
        out = self.pooling(out)

        if self.extracted_embedding == "far":
            assert self.layer6 is not None
            xvector = self.layer6.affine(out)
        elif self.extracted_embedding == "near_affine":
            out = self.auto(self.layer6, out)
            xvector = self.layer7.affine(out)
        elif self.extracted_embedding == "near":
            out = self.auto(self.layer6, out)
            xvector = self.layer7(out)

        return xvector

    def get_warmR_T(T_0, T_mult, epoch):
        n = int(math.log(max(0.05, (epoch / T_0 * (T_mult - 1) + 1)), T_mult))
        T_cur = epoch - T_0 * (T_mult ** n - 1) / (T_mult - 1)
        T_i = T_0 * T_mult ** (n)
        return T_cur, T_i

    def compute_decay_value(self, start, end, T_cur, T_i):
        # Linear decay in every cycle time.
        return start - (start - end)/(T_i-1) * (T_cur % T_i)

    def step(self, epoch, this_iter, epoch_batchs):
        # Heated up for t and s.
        # Decay for margin and dropout p.
        if self.use_step:
            if self.step_params["m"]:
                current_postion = epoch * epoch_batchs + this_iter
                lambda_factor = max(self.step_params["lambda_0"],
                                    self.step_params["lambda_b"]*(1+self.step_params["gamma"]*current_postion)**(-self.step_params["alpha"]))
                self.loss.step(lambda_factor)

            if self.step_params["T"] is not None and (self.step_params["t"] or self.step_params["p"]):
                T_cur, T_i = get_warmR_T(*self.step_params["T"], epoch)
                T_cur = T_cur*epoch_batchs + this_iter
                T_i = T_i * epoch_batchs

            if self.step_params["t"]:
                self.loss.t = self.compute_decay_value(*self.step_params["t_tuple"], T_cur, T_i)

            if self.step_params["p"]:
                self.aug_dropout.p = self.compute_decay_value(*self.step_params["p_tuple"], T_cur, T_i)

            if self.step_params["s"]:
                self.loss.s = self.step_params["s_tuple"][self.step_params["s_list"][epoch]]


if __name__ == '__main__':
    # Input size: batch_size * seq_len * feat_dim
    x = torch.zeros(128, 80, 200)
    model = ECAPA_TDNN(inputs_dim=80, num_targets=5994, channels=512, emb_dim=192)
    # out = model(x)
    print(model)
    # print(out.shape)    # should be [2, 192]

    import numpy as np
    print(np.sum([p.numel() for p in model.parameters()]).item())
