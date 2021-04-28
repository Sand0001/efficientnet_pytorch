import torch
from torch import nn

class ClassifyLoss(nn.Module):
    def __init__(self,training = False):
        super().__init__()
        if training:

            self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        else:
            self.ce_loss = nn.CrossEntropyLoss()

    def forward(self,output,label):   #TODO: 需要调试
        return self.ce_loss(output,label)


class SnapMixLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, criterion, outputs, ya, yb, lam_a, lam_b):
        loss_a = criterion(outputs, ya)
        loss_b = criterion(outputs, yb)
        loss = torch.mean(loss_a * lam_a + loss_b * lam_b)
        return loss


if __name__ == '__main__':
    import torch
    import torch.nn as nn

    x_input = torch.randn(3, 3)  # 随机生成输入
    print('x_input:\n', x_input)
    y_target = torch.tensor([1, 2, 0])  # 设置输出具体值 print('y_target\n',y_target)

    # # 计算输入softmax，此时可以看到每一行加到一起结果都是1
    # softmax_func = nn.Softmax(dim=1)
    # soft_output = softmax_func(x_input)
    # print('soft_output:\n', soft_output)
    #
    # # 在softmax的基础上取log
    # log_output = torch.log(soft_output)
    # print('log_output:\n', log_output)


    # 直接使用pytorch中的loss_func=nn.CrossEntropyLoss()看与经过NLLLoss的计算是不是一样
    crossentropyloss = nn.CrossEntropyLoss()
    crossentropyloss_output = crossentropyloss(x_input, y_target)
    print('crossentropyloss_output:\n', crossentropyloss_output)

    ce = ClassifyLoss()
    d = ce(x_input,y_target)
    print(d)



