#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
S3FD 网络结构。

关键修复：
- fc6 必须使用 dilation=3；
- 原始写法只有 padding=3 会导致结构不符合 S3FD/SSD 的 atrous convolution 设计。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class L2Norm(nn.Module):
    """
    S3FD 特征图 L2 归一化层。
    """
    def __init__(self, n_channels, scale=1.0):
        super(L2Norm, self).__init__()

        self.n_channels = n_channels
        self.scale = scale
        self.eps = 1e-10

        self.weight = nn.Parameter(torch.empty(self.n_channels))
        nn.init.constant_(self.weight, self.scale)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        return x / norm * self.weight.view(1, -1, 1, 1)


class s3fd(nn.Module):
    """
    S3FD 人脸检测网络。
    """
    def __init__(self):
        super(s3fd, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)

        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)

        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)

        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)

        self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, 1, 1)

        # S3FD/SSD 的 fc6 是空洞卷积，padding=3 需要配合 dilation=3。
        self.fc6 = nn.Conv2d(512, 1024, 3, 1, 3, dilation=3)
        self.fc7 = nn.Conv2d(1024, 1024, 1, 1, 0)

        self.conv6_1 = nn.Conv2d(1024, 256, 1, 1, 0)
        self.conv6_2 = nn.Conv2d(256, 512, 3, 2, 1)

        self.conv7_1 = nn.Conv2d(512, 128, 1, 1, 0)
        self.conv7_2 = nn.Conv2d(128, 256, 3, 2, 1)

        self.conv3_3_norm = L2Norm(256, scale=10)
        self.conv4_3_norm = L2Norm(512, scale=8)
        self.conv5_3_norm = L2Norm(512, scale=5)

        self.conv3_3_norm_mbox_conf = nn.Conv2d(256, 4, 3, 1, 1)
        self.conv3_3_norm_mbox_loc = nn.Conv2d(256, 4, 3, 1, 1)

        self.conv4_3_norm_mbox_conf = nn.Conv2d(512, 2, 3, 1, 1)
        self.conv4_3_norm_mbox_loc = nn.Conv2d(512, 4, 3, 1, 1)

        self.conv5_3_norm_mbox_conf = nn.Conv2d(512, 2, 3, 1, 1)
        self.conv5_3_norm_mbox_loc = nn.Conv2d(512, 4, 3, 1, 1)

        self.fc7_mbox_conf = nn.Conv2d(1024, 2, 3, 1, 1)
        self.fc7_mbox_loc = nn.Conv2d(1024, 4, 3, 1, 1)

        self.conv6_2_mbox_conf = nn.Conv2d(512, 2, 3, 1, 1)
        self.conv6_2_mbox_loc = nn.Conv2d(512, 4, 3, 1, 1)

        self.conv7_2_mbox_conf = nn.Conv2d(256, 2, 3, 1, 1)
        self.conv7_2_mbox_loc = nn.Conv2d(256, 4, 3, 1, 1)

    def forward(self, x):
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        f3_3 = h
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        f4_3 = h
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        f5_3 = h
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h))
        ffc7 = h

        h = F.relu(self.conv6_1(h))
        h = F.relu(self.conv6_2(h))
        f6_2 = h

        h = F.relu(self.conv7_1(h))
        h = F.relu(self.conv7_2(h))
        f7_2 = h

        f3_3 = self.conv3_3_norm(f3_3)
        f4_3 = self.conv4_3_norm(f4_3)
        f5_3 = self.conv5_3_norm(f5_3)

        cls1 = self.conv3_3_norm_mbox_conf(f3_3)
        reg1 = self.conv3_3_norm_mbox_loc(f3_3)

        cls2 = self.conv4_3_norm_mbox_conf(f4_3)
        reg2 = self.conv4_3_norm_mbox_loc(f4_3)

        cls3 = self.conv5_3_norm_mbox_conf(f5_3)
        reg3 = self.conv5_3_norm_mbox_loc(f5_3)

        cls4 = self.fc7_mbox_conf(ffc7)
        reg4 = self.fc7_mbox_loc(ffc7)

        cls5 = self.conv6_2_mbox_conf(f6_2)
        reg5 = self.conv6_2_mbox_loc(f6_2)

        cls6 = self.conv7_2_mbox_conf(f7_2)
        reg6 = self.conv7_2_mbox_loc(f7_2)

        # 第一层分类头使用 max-out 合并背景通道。
        chunk = torch.chunk(cls1, 4, dim=1)
        bmax = torch.max(torch.max(chunk[0], chunk[1]), chunk[2])
        cls1 = torch.cat([bmax, chunk[3]], dim=1)

        return [
            cls1,
            reg1,
            cls2,
            reg2,
            cls3,
            reg3,
            cls4,
            reg4,
            cls5,
            reg5,
            cls6,
            reg6,
        ]
