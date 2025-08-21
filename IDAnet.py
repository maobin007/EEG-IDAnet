import os
import sys
from thop import profile
from thop import clever_format

current_path = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(current_path)[0]
sys.path.append(current_path)
sys.path.append(rootPath)

import torch
import torch.nn as nn
from torchinfo import summary
from torchstat import stat
from utils.Transformer_Ecoder_block import TransformEncoder_block
from utils.util import Conv2dWithConstraint, LinearWithConstraint
from utils.Informer_block import InformerEncoder
from utils.Attention_mechanism_block import ECAAttention

# %%
class IDAnet(nn.Module):
    def __init__(self, eeg_chans=22, samples=1000, kerSize=62, F1=16, D1=2, D2=2, kerSize_Tem=[21, 42],r=3, poolSize1=8,
                 poolSize2=8, depth=1, num_heads=3, attn_type='prob', d_ff=5, use_conv=False, attn_dropout=0.1,
                 ddropout=0.1, factor=5, act='elu',dropout_dep=0.1,  n_classes=4):
        super(IDAnet, self).__init__()
        self.F21 = F1 * D1
        self.F22 = F1 * D2
        # ============================= EEGINC branch model =============================
        self.Temporal_conv = nn.Sequential(
            Conv2dWithConstraint(in_channels=1, out_channels=F1, kernel_size=(1, kerSize), stride=1, padding='same', bias=False, max_norm=.5),
            nn.BatchNorm2d(num_features=F1),
        )
        self.TempSpatial_branch1 = nn.Sequential(
            Conv2dWithConstraint(in_channels=F1, out_channels=F1 * D1, kernel_size=(eeg_chans, 1), groups=F1, bias=False,
                                 max_norm=.5),
            nn.BatchNorm2d(num_features=self.F21),
            nn.ELU(),  # inplace=True
            nn.AvgPool2d(kernel_size=(1, poolSize1), stride=(1, poolSize1)),
            nn.Dropout(p=dropout_dep),
            nn.Conv2d(in_channels=self.F21, out_channels=self.F21, kernel_size=(1, kerSize_Tem[0]), stride=1,
                    padding='same', groups=self.F21, bias=False),
            nn.Conv2d(in_channels=self.F21, out_channels=self.F21, kernel_size=(1, 1), stride=1, bias=False),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, poolSize2), stride=(1, poolSize2)),
            nn.BatchNorm2d(num_features=self.F21),
        )
        self.TempSpatial_branch2 = nn.Sequential(
            Conv2dWithConstraint(in_channels=F1, out_channels=F1 * D2, kernel_size=(eeg_chans, 1), groups=F1, bias=False,
                                 max_norm=.5),
            nn.BatchNorm2d(num_features=self.F22),
            nn.ELU(),  # inplace=True
            nn.AvgPool2d(kernel_size=(1, poolSize1), stride=(1, poolSize1)),
            nn.Dropout(p=dropout_dep),
            nn.Conv2d(in_channels=self.F22, out_channels=self.F22, kernel_size=(1, kerSize_Tem[1]), stride=1,
                      padding='same', groups=self.F22, bias=False),
            nn.Conv2d(in_channels=self.F22, out_channels=self.F22, kernel_size=(1, 1), stride=1, bias=False),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, poolSize2), stride=(1, poolSize2)),
            nn.BatchNorm2d(num_features=self.F22),
        )
        #============================================Feature Fusion=================================================
        self.se_block = ECAAttention(ker_size=r)
        self.d_model = samples//poolSize1//poolSize2
        self.infomer = InformerEncoder(num_layers=depth, attn_type=attn_type, d_model=self.d_model, num_heads=num_heads, d_ff=d_ff,use_conv=use_conv,
                                       attn_dropout=attn_dropout, ddropout=ddropout, factor=factor, act=act)


        # ============================= Decision Fusion model =============================
        # self.flatten_eeg = nn.Flatten()
        self.liner_eeg = LinearWithConstraint(in_features= (self.F21 + self.F22) * self.d_model, out_features=n_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        if len(x.shape) is not 4:
            x = torch.unsqueeze(x, 1)
        x = self.Temporal_conv(x)
        x1 = self.TempSpatial_branch1(x)
        x2 = self.TempSpatial_branch2(x)
        x = torch.cat([x1, x2], dim=1)
        # weight = self.se_block(x)
        # x = x + weight
        x = self.se_block(x)
        # # print(x.shape)
        x,_ = self.infomer(torch.squeeze(x)) # [64, 15]
        x = torch.flatten(x, start_dim=1)
        x = self.liner_eeg(x)
        out = self.softmax(x)
        return out


# # %%
#============================ Initialization parameters ============================###
channels = 22
samples = 1000

###============================ main function ============================###
def main():
    input = torch.randn(32, 1, channels, samples)
    print(input.shape)
    model = IDAnet(eeg_chans=22, samples=1000, kerSize=64, F1=16, D1=2, D2=3,r=3,kerSize_Tem=[21, 48], poolSize1=8,
                 poolSize2=8,depth=1, num_heads=3, attn_type='prob', d_ff=5, use_conv=False, attn_dropout=0.1,
                 ddropout=0.1, factor=5, act='elu',dropout_dep=0.1,  n_classes=4)
    out = model(input)
    flops, params = profile(model, inputs=(input))
    flops, params = clever_format([flops, params], "%.3f")
    print(f'FLOPS: {flops:.2f}, Params: {params:.2f}')
    print('===============================================================')
    print('out', out.shape)
    # print('attention_scores', attention_scores.shape)
    print('model', model)
    # summary(model=model, input_size=(1,1,channels,samples), device="cpu")
    # stat(model, (1, channels, samples))
    print(sum(p.numel() for p in model.parameters()))


if __name__ == "__main__":
    main()
#
