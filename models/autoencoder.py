import torch
from torch.nn import Module
import models.diffusion as diffusion
from models.diffusion import VarianceSchedule, D2MP_OB
import numpy as np

class D2MP(Module):
    def __init__(self, config, encoder=None, device="cuda"):
        super().__init__()
        self.config = config
        self.device = device
        self.encoder = encoder

        # 扩散网络
        self.diffnet = getattr(diffusion, config.diffnet)

        self.diffusion = D2MP_OB(
            # net = self.diffnet(point_dim=2, context_dim=config.encoder_dim, tf_layer=config.tf_layer, residual=False),
            net=self.diffnet(point_dim=4, context_dim=config.encoder_dim, tf_layer=config.tf_layer, residual=False),

            # 方差调度器
            var_sched = VarianceSchedule(
                num_steps=100,
                beta_T=5e-2,
                mode='linear'
            ),
            config=self.config
        )


    """
    generate 方法的目的是 基于条件输入生成目标输出
    conds：条件输入，通常是用于生成目标轨迹或图像的初始信息。它是一个包含多个条件数据的列表。
    sample：生成的样本数，用于生成多少个样本。
    bestof：从生成的样本中选择最好的样本数。
    flexibility：灵活性参数，控制生成的样本的多样性或变化程度。
    ret_traj：是否返回目标轨迹。如果为 True，会返回轨迹；如果为 False，则仅返回目标生成结果。
    img_w 和 img_h：图像的宽度和高度，通常用于归一化条件坐标。
    """

    def generate(self, conds, sample, bestof, flexibility=0.0, ret_traj=False, img_w=None, img_h=None):
        cond_encodeds = []
        for i in range(len(conds)):
            tmp_c = conds[i]
            tmp_c = np.array(tmp_c)

            # 归一化条件坐标
            tmp_c[:, 0::2] = tmp_c[:, 0::2] / img_w
            tmp_c[:, 1::2] = tmp_c[:, 1::2] / img_h

            # 将条件坐标转换为张量
            tmp_conds = torch.tensor(tmp_c, dtype=torch.float)
            if len(tmp_conds) != 5:
                pad_conds = tmp_conds[-1].repeat((5, 1))
                tmp_conds = torch.cat((tmp_conds, pad_conds), dim=0)[:5]
            cond_encodeds.append(tmp_conds.unsqueeze(0))
        cond_encodeds = torch.cat(cond_encodeds)
        cond_encodeds = self.encoder(cond_encodeds)

        # 使用扩散模型（self.diffusion）生成目标输出（例如目标轨迹）。
        # sample 指定生成的样本数，bestof 指定从生成的样本中选择最好的几个
        # flexibility 控制样本的多样性，ret_traj 决定是否返回轨迹。
        track_pred = self.diffusion.sample(cond_encodeds, sample, bestof, flexibility=flexibility, ret_traj=ret_traj)
        return track_pred.cpu().detach().numpy()

    def forward(self, batch):
        cond_encoded = self.encoder(batch["condition"]) # B * 64
        loss = self.diffusion(batch["delta_bbox"], cond_encoded)
        return loss