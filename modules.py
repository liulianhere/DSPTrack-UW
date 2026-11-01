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

# class FeatureFusionBlock(nn.Module):
#     def __init__(self, indim, outdim):
#         super().__init__()
#
#         self.block1 = ResBlock(indim, outdim)
#         self.attention = cbam.CBAM(outdim)
#         self.block2 = ResBlock(outdim, outdim)
#
#     def forward(self, x, f16):
#         x = torch.cat([x, f16], 1)
#         x = self.block1(x)
#         r = self.attention(x)
#         x = self.block2(x + r)
#
#         return x


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


# class ASPPConv(nn.Sequential):
#     def __init__(self, in_channels, out_channels, dilation):
#         modules = [
#             nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU()
#         ]
#         super(ASPPConv, self).__init__(*modules)
#
# class ASPPPooling(nn.Sequential):
#     def __init__(self, in_channels, out_channels):
#         super(ASPPPooling, self).__init__(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU())
#
#     def forward(self, x):
#         size = x.shape[-2:]
#         x = super(ASPPPooling, self).forward(x)
#         return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

# class ASPP(nn.Module):
#     def __init__(self, in_channels, out_channels, atrous_rates=[3, 5, 7]):
#         super(ASPP, self).__init__()
#         modules = []
#
#         # Branch 1: 1x1 Conv (保持原分辨率特征)
#         modules.append(nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU()))
#
#         # Branch 2-4: 3x3 Convs with dilation rates (捕捉多尺度特征)
#         # 使用 [3, 5, 7] 避免网格效应并适配 24x24 特征图
#         for rate in atrous_rates:
#             modules.append(ASPPConv(in_channels, out_channels, rate))
#
#         # Branch 5: Global Average Pooling (捕捉图像层级上下文)
#         modules.append(ASPPPooling(in_channels, out_channels))
#
#         self.convs = nn.ModuleList(modules)
#
#         # Final projection
#         # 输入通道数 = out_channels * 5 (5个分支)
#         self.project = nn.Sequential(
#             nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             # Key 生成层通常不需要最后的 ReLU，保持线性输出以允许负值特征匹配
#         )
#
#         self._init_weights()
#
#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight)
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, x):
#         res = []
#         for conv in self.convs:
#             res.append(conv(x))
#         res = torch.cat(res, dim=1)
#         return self.project(res)

class DSConv(nn.Sequential):
    """
    深度可分离卷积：Depthwise Conv + Pointwise Conv
    参数量约为普通 3x3 卷积的 1/9
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(DSConv, self).__init__(
            # 1. Depthwise: 每个通道独立卷积 (Groups = in_channels)
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            # 2. Pointwise: 1x1 卷积融合通道信息
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class StripPooling(nn.Module):
    """
    轻量化条纹池化：内部卷积全部替换为 DSConv
    """

    def __init__(self, in_channels, pool_size, norm_layer=nn.BatchNorm2d):
        super(StripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
        self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))

        inter_channels = int(in_channels / 4)

        # 1x1 卷积本来就很轻，不需要改
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                     norm_layer(inter_channels),
                                     nn.ReLU(True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                     norm_layer(inter_channels),
                                     nn.ReLU(True))

        # 将所有 3x3 卷积替换为 DSConv
        self.conv2_0 = DSConv(inter_channels, inter_channels, kernel_size=3, padding=1)
        self.conv2_1 = DSConv(inter_channels, inter_channels, kernel_size=3, padding=1)
        self.conv2_2 = DSConv(inter_channels, inter_channels, kernel_size=3, padding=1)

        # 针对非对称卷积 (1x3, 3x1)，也可以用 Groups 分组卷积来轻量化
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), groups=inter_channels, bias=False),
            # Depthwise-like
            norm_layer(inter_channels))
        self.conv2_4 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), groups=inter_channels, bias=False),
            # Depthwise-like
            norm_layer(inter_channels))

        self.conv2_5 = DSConv(inter_channels, inter_channels, kernel_size=3, padding=1)
        self.conv2_6 = DSConv(inter_channels, inter_channels, kernel_size=3, padding=1)

        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels * 2, in_channels, 1, bias=False),
                                   norm_layer(in_channels))

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x2_1 = self.conv2_0(x1)
        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), (h, w), mode='bilinear', align_corners=False)
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), (h, w), mode='bilinear', align_corners=False)
        x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), (h, w), mode='bilinear', align_corners=False)
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), (h, w), mode='bilinear', align_corners=False)
        x1 = self.conv2_5(F.relu_(x2_1 + x2_2 + x2_3))
        x2 = self.conv2_6(F.relu_(x2_5 + x2_4))
        out = self.conv3(torch.cat([x1, x2], dim=1))
        return F.relu_(x + out)


class ASPPDSConv(nn.Module):
    """
    带空洞的深度可分离卷积
    """

    def __init__(self, in_channels, out_channels, dilation):
        super(ASPPDSConv, self).__init__()
        self.block = nn.Sequential(
            # Depthwise (with dilation)
            nn.Conv2d(in_channels, in_channels, 3, padding=dilation, dilation=dilation, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # Pointwise
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class StripASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates=[3, 5, 7]):
        super(StripASPP, self).__init__()
        modules = []

        # Branch 1: 1x1 Conv
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        # Branch 2-4: DSConv + Dilation
        for rate in atrous_rates:
            modules.append(ASPPDSConv(in_channels, out_channels, rate))

        self.convs = nn.ModuleList(modules)

        # Branch 5: Strip Pooling
        self.strip_pool = StripPooling(in_channels, (20, 12))

        # ------------------------------------------------------------------
        # ASPP 融合层
        # ------------------------------------------------------------------
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels + in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
            # 注意：最后不加 ReLU，保持线性，为了和 Shortcut 相加
        )

        # ------------------------------------------------------------------
        # 全局残差捷径 (Global Residual Shortcut)
        # ------------------------------------------------------------------
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # ------------------------------------------------------------------
        # [关键 Trick] Zero-Init 零初始化
        # 将 ASPP 分支最后的 BN 权重初始化为 0。
        # 效果：训练初始阶段，Output = 0 * ASPP + 1 * Shortcut
        # 这意味着模型一开始等价于只用了一个简单的 1x1 卷积，
        # 随着训练进行，ASPP 分支的权重会逐渐增加，慢慢引入多尺度信息。
        # 这能极大加速 Loss 下降。
        # ------------------------------------------------------------------
        if isinstance(self.project[1], nn.BatchNorm2d):
            nn.init.constant_(self.project[1].weight, 0)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))

        strip_feat = self.strip_pool(x)
        res.append(strip_feat)

        feat = torch.cat(res, dim=1)

        # ASPP 路径
        aspp_out = self.project(feat)

        # Shortcut 路径
        shortcut_out = self.shortcut(x)

        # 融合：Residual Connection
        return F.relu_(aspp_out + shortcut_out)


class KeyProjection(nn.Module):
    def __init__(self, indim, keydim):
        super().__init__()
        # indim: 768, keydim: 64
        # 针对 384x384 输入 (即 24x24 特征图)，使用 rates=[3, 5, 7]
        self.aspp = StripASPP(indim, keydim, atrous_rates=[3, 5, 7])

    def forward(self, x):
        return self.aspp(x)

# class KeyProjection(nn.Module):
#     def __init__(self, indim, keydim):
#         super().__init__()
#         self.key_proj = nn.Conv2d(indim, keydim, kernel_size=3, padding=1)
#
#         nn.init.orthogonal_(self.key_proj.weight.data)
#         nn.init.zeros_(self.key_proj.bias.data)
#
#     def forward(self, x):
#         return self.key_proj(x)
