import torch
from loss_funcs.adv import *

class WDLoss(nn.Module):
    """
    直接重写adv.py 中 的AdversarialLoss类，增加梯度惩罚项gp，实现W-gp
    """
    def __init__(self, gamma=1.0, max_iter=1000, use_lambda_scheduler=True, mu=10, **kwargs):
        super(WDLoss, self).__init__()
        self.domain_classifier = Discriminator()
        self.use_lambda_scheduler = use_lambda_scheduler
        self.mu_gp = mu
        if self.use_lambda_scheduler:
            self.lambda_scheduler = LambdaSheduler(gamma, max_iter)

    def forward(self, source, target):
        lamb = 1.0
        if self.use_lambda_scheduler:
            lamb = self.lambda_scheduler.lamb()
            self.lambda_scheduler.step()
        source_loss = self.get_WD_result(source, lamb)
        target_loss = self.get_WD_result(target, lamb)
        gp = self.gradient_penalty(Discriminator, source, target)
        # 原作者乘了一个0.5，是因为他将目标域和源域的损失分开计算了，估计不会产生很大的影响
        adv_loss = source_loss - target_loss - self.mu_gp * gp
        return adv_loss

    def get_WD_result(self, x, lamb=1.0):
        x = ReverseLayerF.apply(x, lamb)
        domain_pred = self.domain_classifier(x)
        # 实际上Wassersein距离的计算只需要计算源域和目标域输出值相减即可，因此将函数替换为平均值mean
        loss_adv = torch.mean(domain_pred)
        return loss_adv

    def gradient_penalty(self, discriminator, batch_s, batch_t):
        batchsz = batch_s.shape[0]
        device = batch_s.device

        # [b, 1, 1]
        t = torch.rand([batchsz, 1], dtype=torch.float32, device=device)
        # [b, 1, 1] => [b, w, c] (assuming w and c are the dimensions of batch_s)
        t = t.expand_as(batch_s)

        interpolate = t * batch_s + (1 - t) * batch_t
        # interpolate = torch.unsqueeze(interpolate[0, :], 0)   # Assuming you want the first channel

        # Compute gradients
        d_interpolate_logits = self.domain_classifier(interpolate)
        d_interpolate_logits = d_interpolate_logits.mean()  # To scalar for gradient computation
        grads = torch.autograd.grad(outputs=d_interpolate_logits, inputs=interpolate,
                                    grad_outputs=torch.ones_like(d_interpolate_logits, device=device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

        # [b, w, c] => [b, -1] (but since we took only the first channel, it's actually [b, w])
        grads = grads.view(grads.size(0), -1)
        gp = torch.norm(grads, p=2, dim=1)
        gp = torch.mean((gp - 1) ** 2)

        return gp