import model as gd_model
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
import os
import torch.backends.cudnn as cudnn
from tqdm import tqdm


def generate_paras(anno_path):
    root = ET.parse(anno_path).getroot()
    anno = {}
    anno['xmax'] = float(root.find("./bndbox/xmax").text)  # 4
    anno['xmin'] = float(root.find("./bndbox/xmin").text)  # 5
    anno['ymax'] = float(root.find("./bndbox/ymax").text)  # 6
    anno['ymin'] = float(root.find("./bndbox/ymin").text)  # 7
    anno['w'] = float(root.find("./bndbox/w").text)  # 8
    anno['h'] = float(root.find("./bndbox/h").text)  # 9
    anno['paras'] = np.array([anno['xmax'], anno['xmin'], anno['ymax'], anno['ymin'], anno['w'], anno['h']], dtype=np.float32)
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
label_test_path = r"./data/labeled_data/test"
label_xml_path = r"./data/labeled_data/xml"
test_batch_size = 100
normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
transform = T.Compose([
    T.Resize([64, 64]),
    T.ToTensor(),
    normalize, ])

test_dataset = OurDataset(label_test_path, label_xml_path, transform)
test_dataloader = DataLoader(test_dataset, test_batch_size, shuffle=True, drop_last=False, num_workers=0)
G = gd_model._G()
D = gd_model._D(9)
print(D)
D = torch.nn.DataParallel(D).cuda()
G = torch.nn.DataParallel(G).cuda()
cudnn.benchmark = True
D.load_state_dict(torch.load(r'./pth/parasbest1true.pth'))
T = gd_model.Train(G, D, None, None)
test_acc_top1 = 0.0
test_acc_top2 = 0.0
test_acc_top3 = 0.0
test_paras_loss = 0.0
with tqdm(total=len(test_dataloader)) as pbar:
    for x, y, para in test_dataloader:
        x, y, para = x.cuda(), y.cuda(), para.cuda()
        top1acc, top2acc, top3acc, parasloss = T.test(x, y, para)
        test_acc_top1 += top1acc
        test_acc_top2 += top2acc
        test_acc_top3 += top3acc
        test_paras_loss += parasloss
        pbar.set_postfix(**{'top1': top1acc, 'top2': top2acc, 'top3': top3acc, 'paras_loss': parasloss})
        pbar.update()
test_acc_top1 /= len(test_dataloader)
test_acc_top2 /= len(test_dataloader)
test_acc_top3 /= len(test_dataloader)
test_paras_loss /= len(test_dataloader)
print(f'test_acc_top1 :{test_acc_top1} test_acc_top2 :{test_acc_top2} test_acc_top3 :{test_acc_top3} test_paras :{test_paras_loss}')
