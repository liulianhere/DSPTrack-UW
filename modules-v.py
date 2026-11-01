import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops  # [新增] 需要引入 torchvision.ops 支持可变形卷积
from torchvision import models

from lib.models.dsp.resnet import resnet18
from lib.models.dsp import cbam


# =========================================================================
#  [新增] Part 1: Coordinate Attention (替代 CBAM 的 Channel Attention)
#  优势：在进行通道加权的同时，保留了 X/Y 方向的位置信息
# =========================================================================
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()  # 相比 ReLU，Hardswish 在轻量级网络中表现更好

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # 分解池化
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # 融合编码
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # 切分
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv_h(x_h)  # [N, oup, H, 1]
        x_w = self.conv_w(x_w)  # [N, oup, 1, W]

        # 生成带位置信息的通道权重
        a_h = x_h.sigmoid()
        a_w = x_w.sigmoid()

        return identity * a_h * a_w


# =========================================================================
#  [新增] Part 2: Deformable Spatial Attention (替代 CBAM 的 Spatial Attention)
#  优势：自适应形变，完美贴合水下非刚性目标（如游动的鱼）
# =========================================================================
class DeformableSpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(DeformableSpatialAttention, self).__init__()

        self.kernel_size = kernel_size
        padding = kernel_size // 2

        # 输入是 MaxPool + AvgPool 后的 2 通道特征
        # 1. 偏移量生成器 (Offset Generator)
        # 输出通道 = 2 * k * k (对应 x, y 两个方向的偏移)
        self.offset_conv = nn.Conv2d(2, 2 * kernel_size * kernel_size, kernel_size=kernel_size, padding=padding)

        # 2. 可变形卷积 (Deformable Conv)
        # 将 2 通道特征映射为 1 通道 Attention Map
        self.dcn = ops.DeformConv2d(2, 1, kernel_size=kernel_size, padding=padding)

        self.sigmoid = nn.Sigmoid()

        # 初始化：使初始偏移接近 0，保证训练稳定性
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)

    def forward(self, x):
        # 1. 通道压缩 (Channel Compression)
        # 沿通道轴做 AvgPool 和 MaxPool
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_compress = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)

        # 2. 预测偏移量 (Predict Offsets)
        # 根据图像的局部特征，决定卷积核怎么"变形"
        offset = self.offset_conv(x_compress)

        # 3. 应用可变形卷积生成注意力图
        # 输入是压缩后的特征，利用偏移量进行采样
        out = self.dcn(x_compress, offset)

        # 4. 空间加权
        return x * self.sigmoid(out)


# =========================================================================
#  [新增] Part 3: Hybrid Deformable CBAM (组合模块)
#  结构：Input -> [Coordinate Attention] -> [Deformable Spatial Attention] -> Output
# =========================================================================
class HybridDeformableCBAM(nn.Module):
    def __init__(self, channels):
        super(HybridDeformableCBAM, self).__init__()
        # 1. 通道注意力：使用 CA
        self.ca = CoordAtt(channels, channels)
        # 2. 空间注意力：使用 Deformable Conv
        self.dsa = DeformableSpatialAttention(kernel_size=3)

    def forward(self, x):
        x = self.ca(x)  # 先修正通道和坐标
        x = self.dsa(x)  # 再修正空间形状
        return x


# =========================================================================
#  原始代码部分 (ResBlock 和 FeatureFusionBlock 修改)
# =========================================================================

class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)

        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class FeatureFusionBlock(nn.Module):
    def __init__(self, indim, outdim):
        super().__init__()

        self.block1 = ResBlock(indim, outdim)

        # --- [修改] 使用 HybridDeformableCBAM 替换原始 CBAM ---
        # self.attention = cbam.CBAM(outdim)
        self.attention = HybridDeformableCBAM(outdim)
        # ---------------------------------------------------

        self.block2 = ResBlock(outdim, outdim)

    def forward(self, x, f16):
        x = torch.cat([x, f16], 1)
        x = self.block1(x)

        # HybridDeformableCBAM 内部已经做了 x * scale 的操作
        # 这里的 r 就是加权后的特征 (Attention-Enhanced Features)
        r = self.attention(x)

        # 保持残差结构：Block2(Original + Enhanced)
        x = self.block2(x + r)

        return x


class HistoricalPromptEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = resnet18(pretrained=True, extra_chan=1)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 1/4, 64
        self.layer2 = resnet.layer2  # 1/8, 128
        self.layer3 = resnet.layer3  # 1/16, 256

        self.fuser = FeatureFusionBlock(768 + 256, 384)

    def forward(self, image, key_f16, mask):
        # key_f16 is the feature from the key encoder

        f = torch.cat([image, mask], 1)

        x = self.conv1(f)
        x = self.bn1(x)
        x = self.relu(x)  # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        x = self.layer1(x)  # 1/4, 64
        x = self.layer2(x)  # 1/8, 128
        x = self.layer3(x)  # 1/16, 256
        x = self.fuser(x, key_f16)

        return x


class UpsampleBlock(nn.Module):
    def __init__(self, skip_c, up_c, out_c, scale_factor=2):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_c, up_c, kernel_size=3, padding=1)
        self.out_conv = ResBlock(up_c, out_c)
        self.scale_factor = scale_factor

    def forward(self, skip_f, up_f):
        x = self.skip_conv(skip_f)
        x = x + F.interpolate(up_f, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        x = self.out_conv(x)
        return x


class KeyProjection(nn.Module):
    def __init__(self, indim, keydim):
        super().__init__()
        self.key_proj = nn.Conv2d(indim, keydim, kernel_size=3, padding=1)

        nn.init.orthogonal_(self.key_proj.weight.data)
        nn.init.zeros_(self.key_proj.bias.data)

    def forward(self, x):
        return self.key_proj(x)