import torch
import model as gd_model
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
import torchvision
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET
from tensorboardX import SummaryWriter
import os


def generate_paras(anno_path):
    root = ET.parse(anno_path).getroot()
    anno = {}
    anno['xmax'] = float(root.find("./bndbox/xmax").text)  # 4
    anno['xmin'] = float(root.find("./bndbox/xmin").text)  # 5
    anno['ymax'] = float(root.find("./bndbox/ymax").text)  # 6
    anno['ymin'] = float(root.find("./bndbox/ymin").text)  # 7
    anno['w'] = float(root.find("./bndbox/w").text)  # 8
    anno['h'] = float(root.find("./bndbox/h").text)  # 9
    anno['paras'] = np.array([anno['xmax'], anno['xmin'], anno['ymax'],
                              anno['ymin'], anno['w'], anno['h']], dtype=np.float32)
    return anno['paras']


class OurDataset(Dataset):
    def __init__(self, img_paths, xml_paths, transforms=None):
        self.transforms = transforms
        self.paths = img_paths
        self.paras = xml_paths
        self.img_list = []
        for root, dirs, files in os.walk(img_paths):
            image_list2 = os.listdir(root)
            for img in image_list2:
                if img.endswith('.jpg'):
                    a = os.path.join(root, img)
                    b = a.replace('\\', '/')
                    self.img_list.append(b)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        image = Image.open(img_path)
        path_spl = img_path.split('/')
        sample_class = int(path_spl[-2])
        pic_name = path_spl[-1]
        pic_name_spl = pic_name.split('.')
        paras_path = os.path.join(self.paras, pic_name_spl[0] + ".xml")
        sample_paras = generate_paras(paras_path)
        if self.transforms:
            image = self.transforms(image)
        return image, sample_class, sample_paras


# Initialize paths and training parameters
writer = SummaryWriter(log_dir="./log")
label_train_path = r"./data/labeled_data/train"
label_test_path = r"./data/labeled_data/test"
label_xml_path = r"./data/labeled_data/xml"
unlabel_path = r"./data/uldata"
epochs = 200
train_batch_size = 100
test_batch_size = 100
my_lr = 0.003
seed = 1
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.Resize([64, 64]),
    transforms.ToTensor(),
    normalize, ])
train_dataset = OurDataset(
    label_train_path, label_xml_path, transforms=transform)
test_dataset = OurDataset(
    label_test_path, label_xml_path, transforms=transform)
un_label = torchvision.datasets.ImageFolder(unlabel_path, transform)
la_train_data_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=True, num_workers=0)
la_test_data_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=test_batch_size, shuffle=True, drop_last=True, num_workers=0)
unl_data_loader = torch.utils.data.DataLoader(
    un_label, batch_size=train_batch_size, shuffle=True, drop_last=True, num_workers=0)
unl_data_loader2 = unl_data_loader

# Create a network
noise22 = torch.randn(100, 100, 1, 1)
G = gd_model._G()
D = gd_model._D(9)
print(D)
D = torch.nn.DataParallel(D).cuda()
G = torch.nn.DataParallel(G).cuda()
cudnn.benchmark = True
# D.apply(gd_model.weights_init)

# Load model weights
# D.load_state_dict(torch.load(""))

G.apply(gd_model.weights_init)
print('G params: %.2fM,D params: %.2fM' % (
    sum(p.numel() for p in G.parameters()) / 1000000.0, sum(p.numel() for p in D.parameters()) / 1000000.0))

# Define the optimizer
optimizerD = optim.Adam(D.parameters(), lr=my_lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(G.parameters(), lr=my_lr, betas=(0.5, 0.999))
T = gd_model.Train(G, D, optimizerG, optimizerD)
train_batch_time = 0
best_acc1 = 0.0
best_acc2 = 0.0
best_acc3 = 0.0
best_paras = 100000.0

len_l_loader = len(la_train_data_loader)
len_ul_loader = len(unl_data_loader)
print("len_labeled_dataloader:", len_l_loader)
print("len_unlabeled_dataloader:", len_ul_loader)

label_repate_time = 0
single_label_acc = 0.0
for epoch in range(epochs):
    total_lab, total_unlab, total_train_acc, total_gen, total_paras = 0.0, 0.0, 0.0, 0.0, 0.0
    la_train_data_loader_iter = iter(la_train_data_loader)
    unl_data_loader2_iter = iter(unl_data_loader2)
    for unlabel1, _label1 in unl_data_loader:
        unlabel2, _label2 = next(unl_data_loader2_iter)
        try:
            x, y, paras   = next(la_train_data_loader_iter)
        except StopIteration:
            # test
            test_acc_top1 = 0.0
            test_acc_top2 = 0.0
            test_acc_top3 = 0.0
            test_paras_loss = 0.0
            for testx, testy, testparas in la_test_data_loader:
                top1acc, top2acc, top3acc, parasloss = T.test(
                    testx, testy, testparas)
                test_acc_top1 += top1acc
                test_acc_top2 += top2acc
                test_acc_top3 += top3acc
                test_paras_loss += parasloss
            test_acc_top1 /= len(la_test_data_loader)
            test_acc_top2 /= len(la_test_data_loader)
            test_acc_top3 /= len(la_test_data_loader)
            test_paras_loss /= len(la_test_data_loader)
            if test_acc_top1 > best_acc1:
                best_acc1 = test_acc_top1
                torch.save(D.state_dict(), './pth/parasbest1true.pth')
            if test_acc_top2 > best_acc2:
                best_acc2 = test_acc_top2
                torch.save(D.state_dict(), './pth/parasbest2true.pth')
            if test_acc_top3 > best_acc3:
                best_acc3 = test_acc_top3
                torch.save(D.state_dict(), './pth/parasbest3true.pth')
            if test_paras_loss < best_paras:
                best_paras = test_paras_loss
                torch.save(D.state_dict(), './pth/parasbest1parastrue.pth')
            single_label_acc /= len(la_train_data_loader)
            writer.add_scalar(
                'single/train', single_label_acc, label_repate_time)
            writer.add_scalar('single/acct1', test_acc_top1, label_repate_time)
            writer.add_scalar('single/acct2', test_acc_top2, label_repate_time)
            writer.add_scalar('single/acct3', test_acc_top3, label_repate_time)
            writer.add_scalar('single/parasloss',
                              test_paras_loss, label_repate_time)
            print(
                "Iteration %d,train ac= %.4f,test acc top1 = %.4f,test acc top2 = %.4f,test acc top3 = %.4f,test paras = %.4f,best acc1 = %.4f,best acc2 = %.4f,best acc3 = %.4f,best parasloss = %.4f" % (
                    label_repate_time, single_label_acc, test_acc_top1, test_acc_top2, test_acc_top3, test_paras_loss,
                    best_acc1, best_acc2, best_acc3, best_paras / test_batch_size))
            label_repate_time += 1
            single_label_acc = 0.0
            la_train_data_loader_iter = iter(la_train_data_loader)
            x, y, paras = next(la_train_data_loader_iter)
        # train
        loss_lab, loss_unlab, loss_paras, train_acc = T.train_batch_disc(
            x, y, paras, unlabel1)

        writer.add_scalar('train_iter/loss_lab', loss_lab, train_batch_time)
        writer.add_scalar('train_iter/loss_unlab',
                          loss_unlab, train_batch_time)
        writer.add_scalar('train_iter/loss_paras',
                          loss_paras, train_batch_time)
        writer.add_scalar('train_iter/train_acc', train_acc, train_batch_time)
        train_batch_time += 1

        total_lab += loss_lab

        total_unlab += loss_unlab
        total_train_acc += train_acc
        single_label_acc += train_acc
        total_paras += loss_paras
        loss_gen = T.train_batch_gen(unlabel2)
        G.eval()
        if train_batch_time % 5 == 0:
            torchvision.utils.save_image(G(noise22), str('./pic/' + str(train_batch_time) + '.png'),
                                         nrow=10, padding=2, normalize=True, range=(-1, 1), scale_each=False,
                                         pad_value=0)
        if loss_gen > 1 and epoch > 1:
            loss_gen = T.train_batch_gen(unlabel2)
        total_gen += loss_gen
    total_lab /= len_ul_loader
    total_unlab /= len_ul_loader
    total_train_acc /= len_ul_loader
    total_gen /= len_ul_loader
    total_paras /= len_ul_loader
    test_acc_top1 = 0.0
    test_acc_top2 = 0.0
    test_acc_top3 = 0.0
    for testx, testy, testparas2 in la_test_data_loader:
        top1acc, top2acc, top3acc, parasloss2 = T.test(
            testx, testy, testparas2)
        test_acc_top1 += top1acc
        test_acc_top2 += top2acc
        test_acc_top3 += top3acc
    if (epoch + 1) % 3 == 0:
        my_lr = my_lr * 0.5
        T.update_learning_rate(my_lr)
        print("mylr changes to ", my_lr)
    print("Iteration %d, loss_lab = %.4f, loss_unl = %.4f,loss_gen = %.4f, train acc = %.4f,  paraloss = %.4f" % (
        epoch, total_lab, total_unlab, total_gen, total_train_acc, total_paras))
    writer.add_scalar('train_epoch/loss_supervised', total_lab, epoch)
    writer.add_scalar('train_epoch/un_loss_supervised', total_unlab, epoch)
    writer.add_scalar('train_epoch/gen_loss', total_gen, epoch)
    writer.add_scalar('train_epoch/acc', total_train_acc, epoch)
    writer.add_scalar('train_epoch/paras', total_paras, epoch)
