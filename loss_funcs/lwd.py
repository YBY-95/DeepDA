import torch
from loss_funcs.wd import *

class LWDLoss(WDLoss, LambdaSheduler):
    def __init__(self, num_class, gamma=1.0, max_iter=1000, **kwargs):
        super(LWDLoss, self).__init__(gamma=gamma, max_iter=max_iter, **kwargs)
        self.num_class = num_class
        self.local_classifiers = torch.nn.ModuleList()
        for _ in range(num_class):
            self.local_classifiers.append(Discriminator())

        self.omega, self.lamda, self.mu= 0.5, 1, 10

    def forward(self, source, target, source_logits, target_logits, wd_weights, source_label):
        lamb = self.lamb()
        self.step()
        source_loss = self.get_WD_result(source, lamb)
        target_loss = self.get_WD_result(target, lamb)
        gp = self.gradient_penalty(self.domain_classifier, source, target)
        global_loss = source_loss - target_loss - self.mu_gp * gp
        local_loss = self.get_local_WD_result(source, target, wd_weights, source_label)
        adv_loss = self.omega*global_loss + (1-self.omega)*local_loss -self.mu*gp

        return adv_loss

    def get_local_WD_result(self, source, target, wd_weight, source_label):
        batch_size = source.shape[0]
        sign_s = self.local_sample_divide(source, source_label, self.num_class)
        wd_metric = torch.empty([self.num_class, batch_size])
        wd_t = self.domain_classifier(target)
        for i in range(self.num_class):
            wd_s = self.domain_classifier(sign_s[i])
            wd = wd_s - wd_t
            wd_metric[i] = torch.squeeze(wd)
        local_wd_loss = torch.mean(torch.sum(wd_weight*torch.transpose(wd_metric.cuda(), 0, 1), 1))

        return local_wd_loss

    def local_sample_divide(self, sample, label, num_class):
        """
        把源域的数据按照类别标签排序，提供规律排布的数据供local损失计算
        """
        divide_sample = []  # 长度为类别数的列表，每个元素包含该batch中对应类别的全部样本
        sample_batch = []
        index = []
        index_len = []

        # 从标签中获取同一类型的数据索引，然后收集同一类型的样本
        for i in range(self.num_class):
            index = torch.where(label == i)[0]
            sample_same_class = sample[index]
            divide_sample.append(sample_same_class)
            index_len.append(len(index))

        # 每个类别的样本数量不可能相同，并且一个batch中没有某个类别样的情况，因此需要一些补零和复制来对齐
        for x in divide_sample:
            if len(x) == 0:
                sample_batch.append(torch.zeros([len(sample), 4096]))
                continue
            duplicate_times = len(sample)//len(x)+1
            sample_long = torch.tile(x, [duplicate_times, 1])
            shuffle_index = torch.randperm(len(sample))
            sample_batch.append(sample_long[:len(sample), :][shuffle_index])

        sample_batch = torch.stack(sample_batch)

        return sample_batch