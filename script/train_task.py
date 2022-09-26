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
from torch.utils.data import DataLoader
import os
import yaml
import torch.nn.functional as F
import logging
import sys

logging.basicConfig(
    # filename=r'./logging.text',
    format='%(asctime)s - %(name)s - %(levelname)s -%(module)s:  %(message)s',
    datefmt='%Y-%m-%d %H-%M-%S',
    level=logging.INFO
)


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


def parse_yaml():
    with open(r'para.yaml', 'r') as file:
        yaml_dict = yaml.load(file, Loader=yaml.FullLoader)
    return yaml_dict


def log_sum_exp(x, axis=1):
    m = torch.max(x, dim=1)[0]
    return torch.log(torch.sum(torch.exp(x - m.unsqueeze(1)), dim=axis)) + m


def train_semi(yaml: dict):
    label_path = yaml['path']['label_train_path']
    label_xml_path = yaml['path']['label_xml_path']
    unlabel_path = yaml['path']['unlabel_path']
    label_test_path = yaml['path']['label_test_path']
    batch_size = yaml['para']['batch_size']
    lr = yaml['para']['lr']
    if torch.cuda.is_available():
        cudnn.benchmark = True
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize([64, 64]),
        transforms.ToTensor(),
        normalize])
    logging.info(f'start loading data...')
    la_dataset = OurDataset(label_path, label_xml_path, transform)
    ul_dataset = torchvision.datasets.ImageFolder(unlabel_path, transform)
    test_dataset = OurDataset(label_test_path, label_xml_path, transform)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True, drop_last=True, num_workers=yaml['para']['num_works'])
    la_dataloader = DataLoader(la_dataset, batch_size, shuffle=True, drop_last=True, num_workers=yaml['para']['num_works'])
    ul_dataloader = DataLoader(ul_dataset, batch_size, shuffle=True, drop_last=True, num_workers=yaml['para']['num_works'])
    logging.info(f'Done! label_data:{len(la_dataset)} unlabel_data:{len(ul_dataset)}')
    logging.info(f'Initialize the network')
    G = gd_model._G()
    D = gd_model._D(yaml['para']['num_classes'])
    logging.info(f'Done!\nStart semi-supervised training...')
    D = torch.nn.DataParallel(D).cuda()
    G = torch.nn.DataParallel(G).cuda()
    G.apply(gd_model.weights_init)
    optimizer_G = optim.Adam(G.parameters(), lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr, betas=(0.5, 0.999))
    T = gd_model.Train(G, D, optimizer_G, optimizer_D)
    ul2_dataloader = ul_dataloader
    ul2_dataloader_iter = iter(ul2_dataloader)
    ul_dataloader_iter = iter(ul_dataloader)
    total_lab, total_unlab, total_train_acc, total_gen, total_paras = 0.0, 0.0, 0.0, 0.0, 0.0
    best_acc1, best_acc2, best_acc3, best_para = 0.0, 0.0, 0.0, 100000.0
    ul_time = 0
    for epoch in range(yaml['para']['epoch']):
        for x, y, paras in la_dataloader:
            try:
                ul1, _ = next(ul_dataloader_iter)
                ul2, _ = next(ul2_dataloader_iter)
            except StopIteration:
                total_train_acc /= len(ul_dataloader)
                total_unlab /= len(ul_dataloader)
                total_gen /= len(ul_dataloader)
                total_paras /= len(ul_dataloader)
                total_lab /= len(ul_dataloader)
                ul_dataloader_iter = iter(ul_dataloader)
                ul2_dataloader_iter = iter(ul2_dataloader)
                ul1, _ = next(ul_dataloader_iter)
                ul2, _ = next(ul2_dataloader_iter)
                ul_time += 1
                logging.info(f'Epoch:{epoch}: loss_lab:%.4f loss_ulab:%.4f loss_gen:%.4f train_acc:%.4f' % (
                    total_lab, total_unlab, total_gen, total_train_acc))
                total_lab, total_unlab, total_train_acc, total_gen, total_paras = 0.0, 0.0, 0.0, 0.0, 0.0
                # if (ul_time + 1) % yaml['para']['lr_step'] == 0:
                #     lr = lr * yaml['para']['lr_factor']
                #     for param_group in optimizer_D.param_groups:
                #         param_group['lr'] = lr
                #     logging.info(f"my_lr changes to {lr}")
            loss_lab, loss_ulab, loss_paras, train_acc = T.train_batch_disc(x, y, paras, ul1)
            total_lab += loss_lab
            total_unlab += loss_ulab
            total_paras += loss_paras
            total_gen += T.train_batch_gen(ul2)
            total_train_acc += train_acc
        # test
        test_acc_top1, test_acc_top2, test_acc_top3, test_paras_loss = 0, 0, 0, 0
        for test_x, test_y, test_para in test_dataloader:
            top1, top2, top3, pa = T.test(test_x, test_y, test_para)
            test_acc_top1 += top1
            test_acc_top2 += top2
            test_acc_top3 += top3
            test_paras_loss += pa
        test_acc_top1 /= len(test_dataloader)
        test_acc_top3 /= len(test_dataloader)
        test_acc_top2 /= len(test_dataloader)
        test_paras_loss /= len(test_dataloader)
        if test_acc_top1 > best_acc1:
            best_acc1 = test_acc_top1
            torch.save(D.state_dict(), r'./pth/semi_acc_top1.pth')
        if test_acc_top2 > best_acc2:
            best_acc2 = test_acc_top2
            torch.save(D.state_dict(), r'./pth/semi_acc_top2.pth')
        if test_acc_top3 > best_acc3:
            best_acc3 = test_acc_top3
            torch.save(D.state_dict(), r'./pth/semi_acc_top3.pth')
        if test_paras_loss < best_para:
            best_para = test_paras_loss
            torch.save(D.state_dict(), r'./pth/semi_paratrue1.pth')
        logging.info(
            f'Epoch:{epoch + 1}: test_acc_top1:%.4f test_acc_top2:%.4f test_acc_top3:%.4f test_para_loss:%.2f best_top1:%.4f best_top2:%.4f best_top3:%.4f best_para:%.2f' % (
                test_acc_top1, test_acc_top2, test_acc_top3, test_paras_loss, best_acc1, best_acc2, best_acc3, best_para / batch_size))
        if (epoch + 1) % yaml['para']['lr_step'] == 0:
            lr = lr * yaml['para']['lr_factor']
            for param_group in optimizer_D.param_groups:
                param_group['lr'] = lr
            logging.info(f"my_lr changes to {lr}")
    logging.info(f'Done!\nTesting semi-supervised model performance...')
    return test_net(yaml, 1)


# Semi-supervised training classification-finetuning
def train_cls_finetine(yaml: dict):
    label_path = yaml['path']['label_train_path']
    label_xml_path = yaml['path']['label_xml_path']
    label_test_path = yaml['path']['label_test_path']
    unlabel_path = yaml['path']['unlabel_path']
    batch_size = yaml['para']['batch_size']
    lr = 0.0001
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize([64, 64]),
        transforms.ToTensor(),
        normalize])
    lab_dataset = OurDataset(label_path, label_xml_path, transform)
    ulab_dataset = torchvision.datasets.ImageFolder(unlabel_path, transform)
    test_dataset = OurDataset(label_test_path, label_xml_path, transform)
    lab_dataloader = DataLoader(lab_dataset, batch_size, shuffle=True, drop_last=True, num_workers=yaml['para']['num_works'])
    ulab_dataloader = DataLoader(ulab_dataset, batch_size, shuffle=True, drop_last=True, num_workers=yaml['para']['num_works'])
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True, drop_last=True, num_workers=yaml['para']['num_works'])
    D = gd_model._D(yaml['para']['num_classes'])
    G = gd_model._G()
    checkpoints_path = r'./pth/semi_acc_top1.pth'
    if os.path.exists(checkpoints_path):
        if torch.cuda.is_available():
            cudnn.benchmark = True
            D = torch.nn.DataParallel(D).cuda()
            G = torch.nn.DataParallel(G).cuda()
            D.load_state_dict(torch.load(checkpoints_path))
    else:
        logging.info(f'Model file not found!')
        sys.exit()
    optimizer_G = optim.Adam(G.parameters(), lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr, betas=(0.5, 0.999))
    ul2_dataloader = ulab_dataloader
    ul_iter = iter(ulab_dataloader)
    ul2_iter = iter(ul2_dataloader)
    total_train_acc = 0.0
    best_acc1, best_acc2, best_acc3 = 0.0, 0.0, 0.0
    for epoch in range(yaml['para']['epoch']):
        D.train()
        G.train()
        for x, y, _ in lab_dataloader:
            try:
                ul1, _ = next(ul_iter)
                ul2, _ = next(ul2_iter)
            except StopIteration:
                ul_iter = iter(ulab_dataloader)
                ul2_iter = iter(ul2_dataloader)
                ul1 = next(ul_iter)
                ul2 = next(ul2_iter)
            # cls
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()
            noise1 = torch.randn(batch_size, 100, 1, 1)
            if torch.cuda.is_available():
                x, y, noise1 = x.cuda(), y.cuda(), noise1.cuda()
                ul1, ul2 = ul1.cuda(), ul2.cuda()
            lab, _ = D(x)
            unlab, _ = D(ul1)
            gen = G(noise1)
            gen_output = D(gen, feature=True)
            loss_lab = torch.mean(torch.mean(log_sum_exp(lab))) - torch.mean(torch.gather(lab, 1, y.unsqueeze(1)))
            loss_unlab = 0.5 * (torch.mean(F.softplus(log_sum_exp(unlab))) - torch.mean(log_sum_exp(unlab)) + torch.mean(
                F.softplus(log_sum_exp(gen_output))))
            total_loss = loss_lab + loss_unlab
            acc = torch.mean((lab.max(1)[1] == y).float())
            total_train_acc += acc
            total_loss.backward()
            optimizer_D.step()
            optimizer_G.step()
            # cls-finetuning
        total_train_acc /= len(lab_dataloader)
        # test
        test_acc_top1, test_acc_top2, test_acc_top3 = 0, 0, 0
        for test_x, test_y, _ in test_dataloader:
            if torch.cuda.is_available():
                test_x, test_y = test_x.cuda(), test_y.cuda()
            test_pc, _ = D(test_x)
            best2 = test_pc.topk(3)[1]
            first = torch.tensor(0).cuda()
            second = torch.tensor(1).cuda()
            third = torch.tensor(2).cuda()
            max1 = best2.index_select(1, first).float()
            max2 = best2.index_select(1, second).float()
            max3 = best2.index_select(1, third).float()
            max1 = max1.reshape(max1.shape[0])
            max2 = max2.reshape(max2.shape[0])
            max3 = max3.reshape(max3.shape[0])
            acc1 = torch.mean((max1 == test_y).float())
            acc2 = acc1 + torch.mean((max2 == test_y).float())
            acc3 = acc2 + torch.mean((max3 == test_y).float())
            test_acc_top1 += acc1
            test_acc_top2 += acc2
            test_acc_top3 += acc3
        test_acc_top1 /= len(test_dataloader)
        test_acc_top2 /= len(test_dataloader)
        test_acc_top3 /= len(test_dataloader)
        if test_acc_top1 > best_acc1:
            best_acc1 = test_acc_top1
            torch.save(D.state_dict(), r'./pth/finetuning_cls_acc_top1.pth')
        if test_acc_top2 > best_acc2:
            best_acc2 = test_acc_top2
            torch.save(D.state_dict(), r'./pth/finetuning_cls_acc_top2.pth')
        if test_acc_top3 > best_acc3:
            best_acc3 = test_acc_top3
            torch.save(D.state_dict(), r'./pth/finetuning_cls_acc_top3.pth')
        logging.info(
            f'Epoch:{epoch + 1}: train_acc:%.4f test_acc_top1:%.4f test_acc_top2:%.4f test_acc_top3:%.4f best_top1:%.4f best_top2:%.4f best_top3:%.4f' % (
                total_train_acc, test_acc_top1, test_acc_top2, test_acc_top3, best_acc1, best_acc2, best_acc3))
        total_train_acc = 0.0
        if (epoch + 1) % 50 == 0:
            lr = lr * 0.1
            for param_group in optimizer_D.param_groups:
                param_group['lr'] = lr
            logging.info(f"my_lr changes to {lr}")
    logging.info(f'Done!\nTesting finetuning_cls_model performance...')
    return best_acc1, best_acc2, best_acc3


def test_net(yaml: dict, flag):
    label_test_path = yaml['path']['label_test_path']
    label_xml_path = yaml['path']['label_xml_path']
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize([64, 64]),
        transforms.ToTensor(),
        normalize])
    ds = OurDataset(label_test_path, label_xml_path, transform)
    dsloader = DataLoader(ds, batch_size=yaml['para']['batch_size'], shuffle=True, drop_last=True, num_workers=yaml['para']['num_works'])
    D = gd_model._D(yaml['para']['num_classes'])
    if torch.cuda.is_available():
        D = torch.nn.DataParallel(D).cuda()
    if flag:
        D.load_state_dict(torch.load(r'./pth/semi_acc_top1.pth'))
    else:
        D.load_state_dict(torch.load(r'./pth/full_acc_top1.pth'))
    D.eval()
    acc1_total, acc2_total, acc3_total, loss_paras = 0, 0, 0, 0
    with torch.no_grad():
        for x, y, paras in dsloader:
            if torch.cuda.is_available():
                x, y, paras = x.cuda(), y.cuda(), paras.cuda()
            output, out_paras = D(x)
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
            loss_paras_test = F.l1_loss(paras, out_paras, reduction="none").sum() / yaml['para']['batch_size']
            acc1_total += acc1
            acc2_total += acc2
            acc3_total += acc3
            loss_paras += loss_paras_test
    acc1_total /= len(dsloader)
    acc2_total /= len(dsloader)
    acc3_total /= len(dsloader)
    loss_paras /= len(dsloader)
    return acc1_total, acc2_total, acc3_total, loss_paras


if __name__ == '__main__':
    yaml_para = parse_yaml()
    seed = yaml_para['para']['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    # It proves that the training accuracy of semi-supervision
    if yaml_para['task'] == 0:
        logging.info('Validate the performance of semi-supervised and fully-supervised training')
        semi_acc1, semi_acc2, semi_acc3, semi_para = train_semi(yaml_para)
        logging.info(f'semi_acc1:{semi_acc1} semi_acc2:{semi_acc2} semi_acc3:{semi_acc3} semi_para:{semi_para}\n\n')
    elif yaml_para['task'] == 1:
        logging.info('Fine-tune the model classification head')
        multi_task_cls_acc1, multi_task_cls_acc2, multi_task_cls_acc3, _ = test_net(yaml_para, 1)
        logging.info(
            f'Multi-task model classification head performance:acc_top1:{multi_task_cls_acc1} acc_top2:{multi_task_cls_acc2} acc_top3:{multi_task_cls_acc3}')
        logging.info('Start fine-tuning the classification head')
        finetune_acc1, finetune_acc2, finetune_acc3 = train_cls_finetine(yaml_para)
        logging.info(f'fine-tuning_acc_top1:{finetune_acc1} fine-tuning_acc_top2:{finetune_acc2} fine-tuning_acc_top3:{finetune_acc3}\n\n')
        logging.info(
            f'Multi-task model classification head performance:acc_top1:{multi_task_cls_acc1} acc_top2:{multi_task_cls_acc2} acc_top3:{multi_task_cls_acc3}')
        logging.info(f'fine-tuning_acc_top1:{finetune_acc1} fine-tuning_acc_top2:{finetune_acc2} fine-tuning_acc_top3:{finetune_acc3}\n\n')
