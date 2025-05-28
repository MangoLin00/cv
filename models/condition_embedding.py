import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat



# 生成一个二维位置编码，使用 正弦和余弦函数 来编码目标位置的空间或时间信息
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, class_token=False):
        x = x.permute(1, 2, 0)

        num_feats = x.shape[1]
        num_pos_feats = num_feats
        mask = torch.zeros(x.shape[0], x.shape[2], device=x.device).to(torch.bool)
        batch = mask.shape[0]
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / num_pos_feats)

        pos_y = y_embed[:, :, None] / dim_t
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        return pos_y


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


# transform 编码器层
# TransformerEncoderLayer 处理后的输出是一个通过自注意力机制和前馈神经网络计算得出的新的序列表示
# 包含了输入序列中的上下文信息以及增强的特征表示，可以用于后续的任务
class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()

        # self_attn: 初始化了一个 多头自注意力层（Multihead Attention）
        # 它是 Transformer 的核心组件。这个层接受输入的序列，并通过自注意力机制计算每个位置的注意力权重。
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model

        # 前馈神经网络中的两层线性层。它们之间通常有一个激活函数和 dropout 层。
        # 前馈神经网络的作用是进一步处理每个位置的特征。
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    # 该方法的作用是将位置编码 pos 添加到输入的张量 tensor 上。
    # 如果没有位置编码（即 pos 为 None），则返回原始的 tensor。
    # 如果存在位置编码，则将位置编码与输入张量相加，从而将空间或时间信息编码到输入中。
    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    # 后向传播 的实现方法，即自注意力和前馈网络的计算顺序是先进行自注意力计算，再进行前馈网络计算
    def forward_post(self,
                     src,
                     src_mask = None,
                     src_key_padding_mask = None,
                     pos = None):

        # 自注意力机制（Self-Attention） 计算：
        # self.self_attn 是一个多头自注意力层（MultiheadAttention），它接受输入张量 src，并计算每个位置（词语、时间步等）之间的注意力权重。
        # src：输入张量，通常形状为 (seq_length, batch_size, d_model)，其中 seq_length 是序列的长度，batch_size 是批量大小，d_model 是每个位置的特征维度。
        # attn_mask：可选的遮蔽掩码，用于在计算注意力时遮蔽某些位置，防止模型关注无关的部分。通常用于处理填充部分或强制某些位置不被关注。
        # key_padding_mask：用于指示输入中哪些位置是填充的，这样模型可以忽略这些填充位置的计算。
        # value=src：在标准的自注意力中，查询（q）、键（k）和值（v）是通过不同的线性变换得到的，但在这里，查询、键和值都使用相同的输入 src。
        # [0]：self.self_attn 返回的是一个元组，第一个元素是经过自注意力计算后的输出张量，所以我们通过 [0] 获取它。
        src2 = self.self_attn(src, src, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]

        # 残差连接（Residual Connection）：
        # 这里通过残差连接将自注意力的输出 src2 加到原始输入 src 上。这种做法有助于防止信息丢失并改善梯度流动。
        # self.dropout1(src2)：对自注意力的输出 src2 应用 dropout，目的是减少过拟合。dropout1 是之前初始化的 dropout 层。
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # 前馈神经网络（Feedforward Neural Network）：
        # self.linear1(src)：首先将输入 src 通过一个线性层 linear1，将其映射到 dim_feedforward 大小的特征空间。这是前馈神经网络的第一层。
        # self.activation(...)：对线性层的输出应用激活函数，通常是 ReLU 或 GELU 等非线性激活函数。激活函数使得网络能够学习复杂的非线性关系。
        # self.dropout(...)：在激活函数后应用 dropout，用于正则化，防止过拟合。
        # self.linear2(...)：将经过激活和 dropout 处理后的输出通过第二个线性层 linear2，将其映射回原始的特征维度 d_model。
        # 结果： src2 是前馈神经网络的输出，它是对输入 src 经过两个线性变换和激活的处理结果。
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    # 前向传播 的实现方法，即先进行层归一化，再进行自注意力和前馈网络计算
    def forward_pre(self, src,
                    src_mask = None,
                    src_key_padding_mask = None,
                    pos = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask = None,
                src_key_padding_mask = None,
                pos = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


# 历史运动嵌入模块
# 该 History_motion_embedding 类的作用是：
# 通过多层 Transformer 编码器从目标的历史运动数据中提取特征。
# 使用正弦/余弦函数生成位置编码，将空间或时间位置信息嵌入到输入数据中。
# 在编码器层中通过 cls_token 聚合目标的全局运动特征。
# 最终输出一个包含历史运动信息的嵌入向量，供后续任务使用。
class History_motion_embedding(nn.Module):
    # d_model：表示嵌入空间的维度，通常用来指定神经网络内部的表示维度。
    # nhead：表示多头自注意力机制的头数，这是 Transformer 中的一个超参数。
    # dim_feedforward：表示前馈神经网络中的隐藏层维度。
    # dropout：用于指定 dropout 正则化的比率，用于防止过拟合。
    # activation：激活函数类型，这里默认为 relu。
    # normalize_before：是否在每次注意力机制前进行归一化（用于深度网络的训练）。
    # pos_type：选择的位置编码类型，这里默认为 'sin'，表示使用正弦/余弦位置编码。
    def __init__(self, d_model=256, nhead=8, dim_feedforward=512, dropout=0.1,
                 activation='relu', normalize_before=False, pos_type='sin'):
        super(History_motion_embedding, self).__init__()
        self.cascade_num = 6

        # self.cls_token 是一个可训练的参数（nn.Parameter），它表示一个特殊的分类标记（[CLS] token）
        # 用于 Transformer 模型中的输入，以便最终生成一个全局表示（例如分类任务的标签或整个输入序列的聚合信息）
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.trca = nn.ModuleList()
        for _ in range(self.cascade_num):

            # self.trca.append(...) 的作用是将每一层 TransformerEncoderLayer 添加到 self.trca 列表中
            # 最终形成由多个编码器层组成的模块
            # 允许输入数据经过多个 Transformer 层逐层抽取特征。
            self.trca.append(TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                   dropout, activation, normalize_before))

        # 这是一个线性层，用来将输入数据的特征维度从 8 映射到目标维度 d_model。
        # 这意味着输入数据的每个样本的特征数是 8，而通过这个线性变换将其映射到更高的维度（通常为 256 或其他适当的维度）。
        self.proj = nn.Linear(8, d_model)

        # 用于生成基于正弦和余弦函数的位置信息，帮助模型学习空间位置或时间顺序。
        if pos_type == 'sin':
            self.pose_encoding = PositionEmbeddingSine(normalize=True)


    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0).to(self.cls_token)
        else:
            x = x.to(self.cls_token)

        q_patch = self.proj(x).permute(1, 0, 2)

        # 通过位置编码生成每个时间步的位置信息。这会将输入的运动数据 q_patch 和位置编码结合在一起
        pos = self.pose_encoding(q_patch).transpose(0, 1)

        # 获取 q_patch 的形状信息，n 是序列的长度，b 是批次的大小，d 是特征维度。
        n, b, d = q_patch.shape

        # 这里通过 repeat 函数将 cls_token 重复 b 次（对于每个样本一个），并将其转置为适合与 q_patch 拼接的格式。
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b).permute(1, 0, 2).contiguous()

        # 将 q_patch（历史运动数据的嵌入）和 cls_tokens（分类标记）在 dim=0 维度上拼接。
        # 这使得每个输入序列（q_patch）的开始都包含一个特殊的分类标记，通常用于提取整个序列的全局表示。
        encoder_patch = torch.cat((q_patch, cls_tokens), dim=0)

        for i in range(self.cascade_num):

            # 每一层 Transformer 编码器都会接收当前的输入（encoder_patch）和位置编码（pos）
            # 然后输出经过自注意力和前馈网络处理后的数据。
            # 每层的输出会被传递到下一层，逐步提取更高层次的特征。
            en_out = self.trca[i](src=encoder_patch, pos=pos)
            encoder_patch = en_out

        out = en_out[0].view(b, 1, d).contiguous()
        return out

