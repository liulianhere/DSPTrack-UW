import os
import math
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.models.dsp.modules import *
import skimage
from thop import profile


def softmax_w_top(x, top):
    values, indices = torch.topk(x, k=top, dim=1)
    x_exp = values.exp_()
    x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
    x.zero_().scatter_(1, indices, x_exp.type(x.dtype))
    return x


# ================= 返璞归真版：保留最原始的自然注意力分布 =================
def visualize_affinity(affinity, query_img=None, H_feat=24, W_feat=24, save_dir='./attention_vis'):
    """
    可视化 affinity 注意力图：固定对比单张模板，呈现原汁原味的红蓝渐变
    """
    os.makedirs(save_dir, exist_ok=True)
        
    B, THW, HW = affinity.shape
    aff = affinity[0] # [THW, HW]
    
    # 1. 计算单帧特征的 Token 数量
    HW_mem = H_feat * W_feat
    
    # 2. 仅保留固定对比逻辑：-1 表示上一张记忆帧，0 表示最初始模板
    t_index = -1 
    
    if t_index == -1:
        aff_target = aff[-HW_mem:, :] 
    else:
        aff_target = aff[t_index * HW_mem : (t_index + 1) * HW_mem, :] 
        
    # 3. 提取当前 Search 图像中最强匹配的分数
    attn_weights, _ = aff_target.max(dim=0) # [HW]
    heatmap_24 = attn_weights.view(H_feat, W_feat).detach().cpu().numpy()
    
    # 4. 基础的线性归一化，不加任何魔改
    heatmap_norm = (heatmap_24 - heatmap_24.min()) / (heatmap_24.max() - heatmap_24.min() + 1e-8)
    
    # JET 伪彩色：自然包含从蓝到红的所有渐变色
    heatmap_color_24 = cv2.applyColorMap(np.uint8(255 * heatmap_norm), cv2.COLORMAP_JET)
    
    try:
        if query_img is not None:
            if isinstance(query_img, torch.Tensor):
                img_np = query_img.cpu().numpy()
            else:
                img_np = query_img.copy()
                
            if img_np.shape[-1] == 3:
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR).astype(np.float32)
            else:
                img_bgr = img_np.astype(np.float32)
                
            H_img, W_img = img_bgr.shape[:2]
            
            # 直接线性拉伸到原图尺寸
            heatmap_resized = cv2.resize(heatmap_color_24, (W_img, H_img), interpolation=cv2.INTER_CUBIC).astype(np.float32)
            heatmap_norm_resized = cv2.resize(heatmap_norm, (W_img, H_img), interpolation=cv2.INTER_LINEAR)
            heatmap_norm_base = np.clip(heatmap_norm_resized, 0.0, 1.0)
            
            # === 【纯净的线性叠加逻辑】 ===
            # 不用任何次方运算，直接基于原始分数分配透明度。
            # 分数最低(0)的蓝色区域透明度为 0.2 (淡淡的蓝色滤镜，看清背景)
            # 分数最高(1)的红色区域透明度为 0.6 (鲜艳但不死板的红色，依然能透出目标纹理)
            alpha = 0.2 + 0.4 * heatmap_norm_base
            # ==============================
            
            alpha = alpha[..., np.newaxis]
            
            overlay_float = img_bgr * (1.0 - alpha) + heatmap_resized * alpha
            overlay = np.clip(overlay_float, 0, 255).astype(np.uint8)
            
            cv2.imwrite(os.path.join(save_dir, "search_overlay.jpg"), overlay)
            # 同时也把纯热力图存下来，方便您直接观察无底图干扰的纯净分布
            cv2.imwrite(os.path.join(save_dir, "search_heatmap_only.jpg"), heatmap_resized.astype(np.uint8))
        else:
            heatmap_resized = cv2.resize(heatmap_color_24, (W_feat*16, H_feat*16), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(save_dir, "search_heatmap_only.jpg"), heatmap_resized)
    except Exception as e:
        print(f"可视化 Search 注意力时出错: {e}")

class HistoricalPromptDecoder(nn.Module):
    def __init__(self, isEval=False):
        super().__init__()

    def set_eval(self, mem_max=150):
        self.topk = 20
        self.CK = None
        self.CV = None
        self.mem_k = None
        self.mem_v = None
        self.mem_max = mem_max
        self.mem_cnt = 0
        self.imgs = None
        self.path = None
        self.seq_name = None
        self.frame_id = None

    def setPath(self, path, seqName, frameId):
        self.path = f"{path}/{seqName}/{frameId}"
        self.seq_name = seqName
        self.frame_id = frameId

    def add_memory(self, key, value, is_temp=False, searchRegionImg=None):
        if searchRegionImg is not None:
            img = torch.from_numpy(searchRegionImg[:, :, ::-1].copy())
            if self.imgs is None:
                self.imgs = img
            else:
                if self.mem_cnt == self.mem_max:
                    W_single = img.shape[1]
                    self.imgs = self.imgs[:, W_single:, :]
                self.imgs = torch.cat([self.imgs, img], dim=1)

        self.temp_k = None
        self.temp_v = None
        key = key.flatten(start_dim=2)
        value = value.flatten(start_dim=2)
        if self.mem_k is None:
            self.mem_k = key
            self.mem_v = value
            self.CK = key.shape[1]
            self.CV = value.shape[1]
            self.mem_cnt = 1
        else:
            if is_temp:
                self.temp_k = key
                self.temp_v = value
            else:
                if self.mem_cnt == self.mem_max:
                    self.mem_k = self.mem_k[:, :, key.shape[2]:]
                    self.mem_v = self.mem_v[:, :, value.shape[2]:]
                    self.mem_cnt -= 1

                self.mem_k = torch.cat([self.mem_k, key], 2)
                self.mem_v = torch.cat([self.mem_v, value], 2)
                self.mem_cnt += 1

    def match_memory(self, qk):
        qk = qk.flatten(start_dim=2)
        if self.temp_k is not None:
            mk = torch.cat([self.mem_k, self.temp_k], 2)
            mv = torch.cat([self.mem_v, self.temp_v], 2)
        else:
            mk = self.mem_k
            mv = self.mem_v
        affinity = self._global_matching(mk, qk)
        readout_mem = torch.bmm(affinity, mv)
        return readout_mem.view(qk.shape[0], self.CV, -1)

    def _global_matching(self, mk, qk):
        B, CK, NE = mk.shape
        a_sq = mk.pow(2).sum(1).unsqueeze(2)
        ab = mk.transpose(1, 2) @ qk
        affinity = (2 * ab - a_sq) / math.sqrt(CK)
        affinity = softmax_w_top(affinity, top=self.topk)
        return affinity

    # 接收新增的 query_img 并传递
    def get_affinity(self, mk, qk, eval=False, visualize=False, query_img=None):
        if not eval:
            mk = mk.flatten(start_dim=2)
        B, CK, THW = mk.shape
        qk = qk.flatten(start_dim=2)

        a_sq = mk.pow(2).sum(1).unsqueeze(2)
        ab = mk.transpose(1, 2) @ qk
        affinity = (2 * ab - a_sq) / math.sqrt(CK)

        maxes = torch.max(affinity, dim=1, keepdim=True)[0]
        x_exp = torch.exp(affinity - maxes)
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        affinity = x_exp / x_exp_sum  # Normalize
        
        if visualize:
            seq = getattr(self, 'seq_name', 'unknown_seq')
            fid = getattr(self, 'frame_id', 'unknown_frame')
            current_save_dir = os.path.join('./attention_vis', str(seq), str(fid))

            HW_feat = qk.shape[2]
            H_feat = int(math.sqrt(HW_feat))
            W_feat = HW_feat // H_feat
            visualize_affinity(affinity, query_img=query_img, H_feat=H_feat, W_feat=W_feat, save_dir=current_save_dir)

        return affinity

    def readout(self, affinity, mv, qv, eval=False):
        if not eval:
            mv = mv.flatten(start_dim=2)
        B, CV, THW = mv.shape
        _, _, H, W = qv.shape
        mem = torch.bmm(mv, affinity)
        mem = mem.view(B, CV, H, W)
        mem_out = torch.cat([mem, qv], dim=1)
        return mem_out


class HistoricalPromptNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = HistoricalPromptEncoder()
        self.decoder = HistoricalPromptDecoder()

    def set_eval(self, mem_max):
        self.decoder.set_eval(mem_max)

    def addMemory(self, addMemKey, addMemValue, searchRegionImg):
        self.decoder.add_memory(addMemKey, addMemValue, searchRegionImg=searchRegionImg)

    def encode(self, frame, kf16, mask, other_mask=None):
        f16 = self.encoder(frame, kf16, mask)
        return f16.unsqueeze(2)

        # 将 visualize 和 query_img 暴露为可控参数

    def eval_decode(self, queryFrame_key, queryFrame_value, visualize=True, query_img=None):
        affinity = self.decoder.get_affinity(self.decoder.mem_k, queryFrame_key, eval=True, visualize=visualize,
                                             query_img=query_img)
        return self.decoder.readout(affinity=affinity, mv=self.decoder.mem_v, qv=queryFrame_value, eval=True)

    def decode(self, queryFrame_key, queryFrame_value, memory_key, memory_value, visualize=False, query_img=None):
        affinity = self.decoder.get_affinity(memory_key, queryFrame_key, visualize=visualize, query_img=query_img)
        return self.decoder.readout(affinity=affinity, mv=memory_value, qv=queryFrame_value)

    def forward(self, mode, *args, **kwargs):
        if mode == 'encode':
            return self.encode(*args, **kwargs)
        elif mode == 'train_decode':
            return self.decode(*args, **kwargs)
        elif mode == 'eval_decode':
            return self.eval_decode(*args, **kwargs)
        else:
            raise NotImplementedError