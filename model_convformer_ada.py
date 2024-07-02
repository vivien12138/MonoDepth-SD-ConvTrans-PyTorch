import math
import torch.nn.functional as F
from torchvision.models.resnet import resnet101
import torchvision.models as models
import torch
import torch.nn as nn
from backbone import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5

#define method for segformer

#define method for FPN
def agg_node(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
    )
def smooth(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
    )
def predict(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
        nn.Sigmoid(),
    )
def upshuffle(in_planes, out_planes, upscale_factor):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes * upscale_factor ** 2, kernel_size=3, stride=1, padding=1),
        nn.PixelShuffle(upscale_factor),
        nn.ReLU()
    )

#define class for segformer
class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x
class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, num_classes=20, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
        super(SegFormerHead, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            c1=embedding_dim * 4,
            c2=embedding_dim,
            k=1,
        )

        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x

class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff

class TBDN(nn.Module):
    """
    TBDN(True Boundary Diffusion Network): Similar Design as Semantic Diffusion Network
    input: feature U to be processed of Lth-layer, it should be a tensor of (BATCHSIZE,CHANNELNUMBER, h, w)
           guided feature V such as semantic  guidance map
    output: Y , it should have the same resolution with U
    """

    def __init__(self, in_channels = 128, out_channels= 128, basic_conv=Conv2d_cd, theta=0.7): #theta = 1 is CDC,basic_conv=Conv2d_gcd for guided cdc
        super(TBDN, self).__init__()

        self.conv1 = nn.Sequential(
            basic_conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels*2, out_channels*2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
    def convg1(self,x,z):
        self.conv_gg = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1,
                              dilation=1, groups=1, bias=False, device = 'cuda')
        guid_out1 = self.conv_gg(torch.mul(x, z))
        guid_out2 = torch.mul(self.conv_gg(x), z)
        guid_out3 = torch.mul(x, self.conv_gg(z))

        [C_out, C_in, kernel_size, kernel_size] = self.conv_gg.weight.shape
        kernel_diff = self.conv_gg.weight.sum(2).sum(2)
        kernel_diff = kernel_diff[:, :, None, None]
        guid_out4 = F.conv2d(input=torch.mul(x, z), weight=kernel_diff, bias=self.conv_gg.bias, stride=self.conv_gg.stride,
                             padding=0, groups=self.conv_gg.groups)
        return guid_out1 - guid_out2 - guid_out3 + guid_out4

    def forward(self, U, V):
        # only CDC
        '''
        x_input = U
        x = self.conv1(x_input)
        Y = torch.cat((U,x),dim=1)
        Y = self.conv2(Y)
        '''
        # V Guided U
        _,_,hu,wu = U.size()
        _,_,hv,wv = V.size()
        if hu!=hv:
            z_input = F.interpolate(V, size=(hu, wu), mode='bilinear', align_corners=True)
        else:
            z_input = V
        x_input = U
        guid_out = self.convg1(x_input, z_input)
        Y = torch.cat((U, guid_out), dim=1)
        Y = self.conv2(Y)

        return Y

class I2D(nn.Module):
    def __init__(self, pretrained_fpn=True, pretrained=False, phi = 'b0', num_classes = 1, fixed_feature_weights = False):
        super(I2D, self).__init__()

        # FPN
        if pretrained_fpn:
            resnet = resnet101(weights=models.ResNet101_Weights.DEFAULT)

        # Freeze those weights
        if fixed_feature_weights:
            for p in resnet.parameters():
                p.requires_grad = False

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = nn.Sequential(resnet.layer1)
        self.layer2 = nn.Sequential(resnet.layer2)
        self.layer3 = nn.Sequential(resnet.layer3)
        self.layer4 = nn.Sequential(resnet.layer4)

        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Aggregate layers
        self.agg1 = agg_node(256, 128)
        self.agg2 = agg_node(256, 128)
        self.agg3 = agg_node(256, 128)
        self.agg4 = agg_node(256, 128)
        self.agg5 = agg_node(256, 128)

        # Aggregate layers for segformer
        self.aggs1 = agg_node(32, 128)
        self.aggs2 = agg_node(64, 128)
        self.aggs3 = agg_node(160, 128)
        self.aggs4 = agg_node(256, 128)

        # Upshuffle layers
        self.up1 = upshuffle(128, 128, 8)
        self.up2 = upshuffle(128, 128, 4)
        self.up3 = upshuffle(128, 128, 2)

        # Depth prediction
        self.predict1 = smooth(512, 128)
        self.predict2 = predict(128, 1)

        #Segformer
        self.in_channels = {
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
            'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
        }[phi]
        self.backbone = {
            'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
            'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
        }[phi](pretrained)
        self.embedding_dim = {
            'b0': 256, 'b1': 256, 'b2': 768,
            'b3': 768, 'b4': 768, 'b5': 768,
        }[phi]
        self.decode_head = SegFormerHead(num_classes, self.in_channels, self.embedding_dim)
        self.tbdn = TBDN()

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, x_L, x_H):
        _, _, H_L, W_L = x_L.size()
        _, _, H_H, W_H = x_H.size()

        # LR Bottom-up
        c1 = self.layer0(x_L)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        # LR Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p4 = self.smooth1(p4)
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p3 = self.smooth2(p3)
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        p2 = self.smooth3(p2)

        #HR  Bottom-up
        c1_h = self.layer0(x_H)
        c2_h = self.layer1(c1_h)
        c3_h= self.layer2(c2_h)
        c4_h = self.layer3(c3_h)
        c5_h = self.layer4(c4_h)
        #HR Top-down
        p5_h = self.toplayer(c5_h)
        p4_h = self._upsample_add(p5_h, self.latlayer1(c4_h))
        p4_h = self.smooth1(p4_h)
        p3_h = self._upsample_add(p4_h, self.latlayer2(c3_h))
        p3_h = self.smooth2(p3_h)
        p2_h = self._upsample_add(p3_h, self.latlayer3(c2_h))
        p2_h = self.smooth3(p2_h)

        # Top-down predict and refine
        a5 = self.agg1(p5)
        d5 = self.up1(a5)
        a4 = self.agg2(p4)
        d4 = self.up2(a4)
        a3 = self.agg3(p3)
        d3 = self.up3(a3)
        d2 = self.agg4(p2)
        _, _, H, W = d2.size()
        vol = torch.cat([F.upsample(d, size=(H, W), mode='bilinear') for d in [d5, d4, d3, d2]], dim=1)
        pred1 = self.predict2(self.predict1(vol))

        #segformer foward
        sf_h = self.backbone.forward(x_H)
        #pred = self.decode_head.forward(sf)

        #HR fuse
        a5_h = self.agg4(p5_h)
        b5_h = self.aggs4(sf_h[3])
        a4_h = self.agg3(p4_h)
        b4_h = self.aggs3(sf_h[2])
        a3_h = self.agg2(p3_h)
        b3_h = self.aggs2(sf_h[1])
        a2_h = self.agg1(p2_h)
        b2_h = self.aggs1(sf_h[0])
        hp5_ = a5_h + b5_h
        hp4_ = a4_h + b4_h
        hp3_ = a3_h + b3_h
        hp2_ = a2_h + b2_h

        ########################################
        # HR upsample
        d5_h = self.up1(hp5_)
        d4_h = self.up2(hp4_)
        d3_h = self.up3(hp3_)
        d2_h = hp2_

        pred_l = pred1

        # HR predict
        _, _, H_h, W_h = d2_h.size()
        vol_h = torch.cat([F.interpolate(d_h, size=(H_h, W_h), mode='bilinear') for d_h in [d5_h, d4_h, d3_h, d2_h]],
                          dim=1)
        pred_h = self.predict2(self.agg5(self.tbdn(self.predict1(vol_h), self.predict1(vol))))
        ##################################################
        return pred_l, pred_h  # img : depth = 4 : 1
