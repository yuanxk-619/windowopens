import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torch.nn.functional as F
import torch.nn.utils.weight_norm as Weight_norm


# load data
def loaddataset(path_train, path_test, path_ul, transform):
    train_label = torchvision.datasets.ImageFolder(path_train, transform)
    test_label = torchvision.datasets.ImageFolder(path_test, transform)
    un_label = torchvision.datasets.ImageFolder(path_ul, transform)
    return train_label, test_label, un_label


class _G(nn.Module):
    def __init__(self):  # inputs (batch,c,w,h)
        super(_G, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),  # 7*7
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),  # 28*28
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # 56*56
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),  # 112*112
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),  # 224*224
            nn.Tanh()
        )

    def forward(self, inputs):
        return self.main(inputs)


class _D(nn.Module):
    def __init__(self, num_classes):
        super(_D, self).__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        layers = list(resnet.children())[:8]
        self.features1 = nn.Sequential(*layers[:6])
        self.features2 = nn.Sequential(*layers[6:])
        self.main2 = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, num_classes))
        self.main3 = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 6))

    def forward(self, inputs, feature=False):
        output = self.features1(inputs)
        output = self.features2(output)
        output = F.relu(output)
        output = nn.AdaptiveAvgPool2d((1, 1))(output)
        if feature:
            return output
        output = output.view(output.shape[0], -1)
        features_paras = self.main3(output)
        features_class = self.main2(output)
        return torch.squeeze(features_class), torch.squeeze(features_paras)


class Train(object):
    def __init__(self, G, D, G_optim, D_optim):
        self.G = G
        self.D = D
        self.G_optim = G_optim
        self.D_optim = D_optim
        self.noise = torch.randn(100, 100, 1, 1)

    def log_sum_exp(self, x, axis=1):
        m = torch.max(x, dim=1)[0]
        return torch.log(torch.sum(torch.exp(x - m.unsqueeze(1)), dim=axis)) + m

    def train_batch_disc(self, x, y, y_paras, x_unlabel):
        noise = torch.randn(y.shape[0], 100, 1, 1)
        if torch.cuda.is_available():
            noise = noise.cuda()
            x = x.cuda()
            x_unlabel = x_unlabel.cuda()
            y = y.cuda()
            y_paras = y_paras.cuda()

        self.D.train()
        self.G.train()

        lab, paras = self.D(x)
        unlab, meiyong = self.D(x_unlabel)
        gen = self.G(noise)
        gen_output, yemeiyong = self.D(gen)
        # loss
        loss_lab = torch.mean(torch.mean(self.log_sum_exp(lab))) - torch.mean(torch.gather(lab, 1, y.unsqueeze(1)))
        loss_unlab = 0.5 * (torch.mean(F.softplus(self.log_sum_exp(unlab))) - torch.mean(self.log_sum_exp(unlab)) + torch.mean(
            F.softplus(self.log_sum_exp(gen_output))))
        loss_paras = F.l1_loss(paras, y_paras, reduction="none").sum(1)
        loss_paras = loss_paras.sum()
        total_loss = loss_lab + loss_unlab + 0.00001 * loss_paras
        # acc
        acc = torch.mean((lab.max(1)[1] == y).float())
        self.D_optim.zero_grad()
        total_loss.backward()
        self.D_optim.step()
        return loss_lab.item(), loss_unlab.item(), loss_paras.item(), acc.item()

    def train_batch_gen(self, x_unlabel):
        noise = torch.randn(x_unlabel.shape[0], 100, 1, 1)
        if torch.cuda.is_available():
            noise = noise.cuda()
            x_unlabel = x_unlabel.cuda()

        self.D.train()
        self.G.train()

        gen_data = self.G(noise)
        output_unl = self.D(x_unlabel, feature=True)
        output_gen = self.D(gen_data, feature=True)

        m1 = torch.mean(output_unl, dim=0)
        m2 = torch.mean(output_gen, dim=0)
        loss_gen = torch.mean(torch.abs(m1 - m2))
        self.G_optim.zero_grad()
        self.D_optim.zero_grad()
        loss_gen.backward()
        self.G_optim.step()
        return loss_gen.item()

    def update_learning_rate(self, lr):
        for param_group in self.G_optim.param_groups:
            param_group['lr'] = lr
        for param_group in self.D_optim.param_groups:
            param_group['lr'] = lr

    def test(self, x, y, paras):
        self.D.eval()
        self.G.eval()
        with torch.no_grad():
            if torch.cuda.is_available():
                x, y, paras = x.cuda(), y.cuda(), paras.cuda()
            output, out_paras = self.D(x)
            best2 = output.topk(3)[1]
            first = torch.tensor(0).cuda()
            second = torch.tensor(1).cuda()
            third = torch.tensor(2).cuda()
            max1 = best2.index_select(1, first).float()
            max2 = best2.index_select(1, second).float()
            max3 = best2.index_select(1, third).float()
            max1 = max1.reshape(max1.shape[0])
            max2 = max2.reshape(max2.shape[0])
            max3 = max3.reshape(max3.shape[0])
            acc1 = torch.mean((max1 == y).float())
            acc2 = acc1 + torch.mean((max2 == y).float())
            acc3 = acc2 + torch.mean((max3 == y).float())
            loss_paras_test = F.l1_loss(paras, out_paras, reduction="none").sum()
        return acc1.item(), acc2.item(), acc3.item(), loss_paras_test.item()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.05)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.05)
        nn.init.constant_(m.bias.data, 0)
