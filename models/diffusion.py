import torch.nn.functional as F
from .common import *

class VarianceSchedule(Module):

    def __init__(self, num_steps, mode='linear',beta_1=1e-4, beta_T=5e-2,cosine_s=8e-3):
        super().__init__()
        assert mode in ('linear', 'cosine')

        """
        num_steps：扩散过程的时间步数（通常是从 1 到 num_steps 的整数）。
        mode：噪声调度的方式，可以是 'linear' 或 'cosine'。这决定了每个时间步的噪声强度（beta）如何变化。
        beta_1 和 beta_T：表示初始噪声强度（beta_1）和最终噪声强度（beta_T）的值。beta 控制每个时间步的噪声强度。
        cosine_s：当选择 cosine 模式时，它是与噪声调度相关的一个常数。
        """
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)
        elif mode == 'cosine':
            timesteps = (
            torch.arange(num_steps + 1) / num_steps + cosine_s
            )
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding
        alphas = 1 - betas
        log_alphas = torch.log(alphas)

        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()
        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)
        # self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alpha_bars))
        # self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alpha_bars - 1))

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas


    """
     结合了位置编码、时间步嵌入和上下文特征的融合，利用多个 MFL 层和 Transformer 编码器进行特征处理和提取。
     最终输出融合后的特征
    """

class HMINet(Module):

    def __init__(self, point_dim=4, context_dim=256, tf_layer=3, residual=False):
        super().__init__()
        self.residual = residual

        # 用于在输入特征中加入位置信息
        self.pos_emb = PositionalEncoding(d_model=2*context_dim, dropout=0.1, max_len=24)
        self.pos_emb2= PositionalEncoding(d_model=context_dim, dropout=0.1, max_len=24)

        # 特征融合层（MFL）用于对输入特征进行融合
        self.concat1 = MFL(4, context_dim // 2, context_dim+3)
        self.concat1_2 = MFL(context_dim // 2, context_dim, context_dim + 3)
        self.concat1_3 = MFL(context_dim, 2 * context_dim, context_dim + 3)

        self.layer = nn.TransformerEncoderLayer(d_model=2*context_dim, nhead=4, dim_feedforward=4*context_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=tf_layer)
        self.layer2 = nn.TransformerEncoderLayer(d_model=context_dim, nhead=4, dim_feedforward=2 * context_dim)
        self.transformer_encoder2 = nn.TransformerEncoder(self.layer2, num_layers=tf_layer)

        # 定义MFL层concat3，将特征降维
        self.concat3 = MFL(2*context_dim,context_dim, context_dim+3)
        self.concat4 = MFL(context_dim,context_dim//2, context_dim+3)
        self.linear = MFL(context_dim//2, 4, context_dim+3)
        #self.linear = nn.Linear(128,2)

    def forward(self, x, beta, context):
        batch_size = x.size(0)
        # beta：时间步信息。
        beta = beta.view(batch_size, 1)          # (B, 1)
        # context：上下文特征
        context = context.view(batch_size, -1)   # (B, F)

        # 时间嵌入和上下文特征融合
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, F+3)

        x = self.concat1_3(ctx_emb, self.concat1_2(ctx_emb, self.concat1(ctx_emb,x)))
        final_emb = x.unsqueeze(0)
        final_emb = self.pos_emb(final_emb)
        trans = self.transformer_encoder(final_emb).permute(1,0,2).squeeze(1)
        trans = self.concat3(ctx_emb, trans)
        final_emb = trans.unsqueeze(0)
        final_emb = self.pos_emb2(final_emb)
        trans = self.transformer_encoder2(final_emb).permute(1, 0, 2).squeeze(1)
        trans = self.concat4(ctx_emb, trans)
        return self.linear(ctx_emb, trans)




class D2MP_OB(Module):

    """
    net: 传入的神经网络模型（例如，self.diffnet），用于处理扩散过程的神经网络。
    var_sched: 方差调度器，用于控制扩散过程中的噪声变化（VarianceSchedule）。
    config: 配置对象，包含一些超参数（例如 eps）。
    self.eps: 从 config 中提取的超参数 eps。
    self.weight: 设置为 True，可能用于在损失计算时选择是否加权。
    """

    def __init__(self, net, var_sched:VarianceSchedule, config):
        super().__init__()
        self.config = config
        self.net = net
        self.var_sched = var_sched
        self.eps = self.config.eps
        self.weight = True

    # 前向扩散过程，向原始数据（x_0）逐步添加噪声，最终使数据变为纯噪声
    def q_sample(self, x_start, noise, t, C):
        time = t.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        x_noisy = x_start + C * time + torch.sqrt(time) * noise
        return x_noisy

    # 这个函数用于根据当前数据 xt（扩散后的数据）预测原始数据 x_0。通过从 xt 中减去噪声和与时间 t 相关的项来实现。
    def pred_x0_from_xt(self, xt, noise, C, t):
        time = t.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        x0 = xt - C * time - torch.sqrt(time) * noise
        return x0

    # 该函数根据当前数据 xt 和噪声 noise 计算常数因子 C，用于在扩散过程中调节噪声的影响
    def pred_C_from_xt(self, xt, noise, t):
        time = t.reshape(noise.shape[0], *((1,) * (len(noise.shape) - 1)))
        C = (xt - torch.sqrt(time) * noise) / (time - 1)
        return C

    # 后向扩散过程，目的是从当前的状态 xt 预测 预测和生成上一个时间步 xt-1 的状态 xt-1，即通过去噪恢复原始数据的过程
    def pred_xtms_from_xt(self, xt, noise, C, t, s):

        # t 和 s 的维度调整，使它们与输入张量 C 的维度匹配。
        # t: 当前时间步，通常是一个从 1 到 T 的值。t.reshape() 将 t 的形状调整为与 C 的形状兼容，确保它在广播过程中能够与其他变量匹配。
        # s: 当前的采样步长，这里将 s 的形状调整为与 C 兼容。s 是一个用于在后向过程中的逐步去噪计算中表示的变量。
        time = t.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        s = s.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))

        # 这一步的目的是生成上一个时间步的 mean 值，表示在去噪过程中恢复出的数据。
        mean = xt + C * (time-s) - C * time - s / torch.sqrt(time) * noise

        # epsilon 是从标准正态分布（均值为 0，方差为 1）中采样得到的噪声，与 mean 形状一致。
        # 这个噪声用于在生成下一个数据时引入随机性。
        epsilon = torch.randn_like(mean, device=xt.device)

        # sigma 计算的是标准差（扩散模型中的噪声强度），它的作用是根据时间步 t 和当前步长 s 计算出当前时间步的噪声水平。
        # 具体而言，sigma 用于控制在恢复过程中引入噪声的幅度。
        # time - s 是在后向扩散过程中的变化量，sigma 就是根据这种变化量计算出的噪声强度。
        sigma = torch.sqrt(s * (time-s) / time)

        # 这一行是扩散模型的核心，生成了 xtms，即根据当前的均值 mean 和噪声 epsilon，加上噪声标准差 sigma 进行调整，得到当前时间步 xt 在前一个时间步的恢复值。
        # xtms 代表当前的去噪数据，经过噪声调整后，预测出下一个时间步t-1的状态。
        xtms = mean + sigma * epsilon
        return xtms

    def forward(self, x_0, context, t=None):
        batch_size, point_dim = x_0.size()
        if t == None:
            t = torch.rand(x_0.shape[0], device=x_0.device) * (1. - self.eps) + self.eps


        # beta 是扩散过程中的噪声调节因子。通过对时间步 t 取对数并除以 4 来计算 beta。
        # 这可以控制噪声的大小，通常在扩散模型中，beta 控制着噪声的加成量。
        beta = t.log() / 4

        # e_rand 是与 x_0 形状相同的随机噪声，使用 torch.randn_like(x_0) 生成。
        e_rand = torch.randn_like(x_0).cuda()  # (B, N, d)
        C = -1 * x_0
        x_noisy = self.q_sample(x_start=x_0, noise=e_rand, t=t, C=C)
        t = t.reshape(-1, 1)

        pred = self.net(x_noisy, beta=beta, context=context)
        C_pred = pred
        noise_pred = (x_noisy - (t - 1) * C_pred) / t.sqrt()
        if not self.weight:
            loss_C = F.smooth_l1_loss(C_pred.view(-1, point_dim), C.view(-1, point_dim), reduction='mean')
            # loss_x0 = F.smooth_l1_loss(x_rec.view(-1, point_dim), x_0.view(-1, point_dim), reduction='mean')
            loss_noise = F.smooth_l1_loss(noise_pred.view(-1, point_dim), e_rand.view(-1, point_dim), reduction='mean')
            loss = 0.5 * loss_C + 0.5 * loss_noise
        else:

            # 根据时间步 t 计算加权系数 simple_weight1 和 simple_weight2。
            # 这些加权系数可以让模型在不同的时间步中对不同的损失项进行不同的加权。
            simple_weight1 = (t ** 2 - t + 1) / t
            simple_weight2 = (t ** 2 - t + 1) / (1 - t + self.eps)

            # simple_weight1 = (t + 1) / t
            # simple_weight2 = (2 - t) / (1 - t + self.eps)

            # 同样使用 L1 损失计算 loss_C 和 loss_noise，但是这次是逐元素计算（reduction='none'）
            # 然后用加权系数 simple_weight1 和 simple_weight2 计算最终损失。
            loss_C = F.smooth_l1_loss(C_pred.view(-1, point_dim), C.view(-1, point_dim), reduction='none')
            # loss_x0 = F.smooth_l1_loss(x_rec.view(-1, point_dim), x_0.view(-1, point_dim), reduction='none')
            loss_noise = F.smooth_l1_loss(noise_pred.view(-1, point_dim), e_rand.view(-1, point_dim), reduction='none')
            loss = simple_weight1 * loss_C + simple_weight2 * loss_noise
            loss = loss.mean()

            # loss = F.smooth_l1_loss(noise_pred.view(-1, point_dim), e_rand.view(-1, point_dim), reduction='mean')

        return loss

    def sample(self, context, sample, bestof, point_dim=4, flexibility=0.0, ret_traj=False):
        """
        sample: 这是函数的入口，表示要进行采样的函数。
        context: 上下文信息，通常是与样本相关的附加信息，如标签或特征。
        sample: 采样次数，即要生成多少个样本。
        bestof: 是否采用“best of”策略（选择最好的结果）。通常在生成过程中，模型会生成多个候选结果，然后选出最优的。
        point_dim: 生成的样本的维度。
        flexibility: 控制生成样本的灵活性，通常与扩散模型的噪声调度和采样方式有关。
        ret_traj: 是否返回采样过程中每个时间步的轨迹。
        """

        traj_list = []
        # context = context.to(self.var_sched.betas.device)
        for i in range(sample):
            batch_size = context.size(0)

            """
            如果 bestof 为 True，则生成一个标准正态分布的随机噪声样本 x_T，作为扩散过程的初始状态。"
            如果 bestof 为 False，则使用全零向量作为初始状态。
            """
            if bestof:
                x_T = torch.randn([batch_size, point_dim]).to(context.device)
            else:
                x_T = torch.zeros([batch_size, point_dim]).to(context.device)

            self.var_sched.num_steps = 1
            traj = {self.var_sched.num_steps: x_T}

            cur_time = torch.ones((batch_size,), device=x_T.device)
            step = 1. / self.var_sched.num_steps
            for t in range(self.var_sched.num_steps, 0, -1):
                s = torch.full((batch_size,), step, device=x_T.device)
                if t == 1:
                    s = cur_time

                x_t = traj[t]
                beta = cur_time.log() / 4
                t_tmp = cur_time.reshape(-1, 1)

                """
                使用网络 self.net 来对当前状态 x_t 进行预测。
                self.net 模型会基于当前状态和上下文信息预测当前时间步的噪声强度或其他相关信息（C_pred）
                """
                pred = self.net(x_t, beta=beta, context=context)
                C_pred = pred
                noise_pred = (x_t - (t_tmp - 1) * C_pred) / t_tmp.sqrt()

                "通过调用 pred_x0_from_xt 方法，从当前时间步 t 恢复出原始的未加噪数据。"
                x0 = self.pred_x0_from_xt(x_t, noise_pred, C_pred, cur_time)
                x0.clamp_(-1., 1.)
                C_pred = -1 * x0

                "x_next 是通过调用 pred_xtms_from_xt 方法，利用当前时间步 t 的状态和噪声，预测下一个时间步 t-1 的状态。"
                x_next = self.pred_xtms_from_xt(x_t, noise_pred, C_pred, cur_time, s)
                cur_time = cur_time - s
                traj[t-1] = x_next.detach()     # Stop gradient and save trajectory.
                if not ret_traj:
                   del traj[t]

            if ret_traj:
                traj_list.append(traj)
            else:
                traj_list.append(traj[0])

        return torch.stack(traj_list)
