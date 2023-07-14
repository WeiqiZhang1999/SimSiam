# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :SSLProject
# @File     :HRFormer
# @Date     :7/13/2023 10:26 AM
# @Author   :Weiqi Zhang
# @Email    :zhang.weiqi.zs9@is.naist.jp
# @Software :PyCharm
-------------------------------------------------
"""
import torch
import torch.nn as nn
from .HRFormerBlock import HighResolutionTransformer
from Network.model.ModelHead.MultiscaleClassificationHead import MultiscaleClassificationHead
import torch.nn.functional as F


class BaseHRFormer(nn.Module):
    def __init__(self, in_channel=1, num_classes=128):
        super(BaseHRFormer, self).__init__()

        self.encoder = HighResolutionTransformer(cfg='hrt_base', input_nc=in_channel)
        self.fuse = MultiscaleClassificationHead(input_nc=sum(self.encoder.output_ncs),
                                                 output_nc=(64 * (2 ** 2)),
                                                 norm_type="group",
                                                 padding_type="reflect")
        self.fc = nn.Linear(self.fuse.output_nc, num_classes)

    def forward(self, x):
        y = self.fuse(self.encoder(x))
        y = F.avg_pool2d(y, kernel_size=y.size()[2:]).view(y.size(0), -1)
        y = self.fc(y)
        return y


if __name__ == '__main__':
    x = torch.randn(1, 1, 256, 128)
    model = BaseHRFormer()
    y = model(x)