# encoding:utf-8
'''
功能：一个图片数据加载函数
'''
import os, csv, glob, random
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image


class Pokemon(Dataset):

    def __init__(self, root, resize, mode):
        super(Pokemon, self).__init__()

        self.root = root
        self.resize = resize

        self.name2label = {}
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):
                continue

            self.name2label[name] = len(self.name2label.keys())
        print(self.name2label)
        # image_path, label
        self.images, self.labels = self.load_csv('images.csv')

        if mode == 'train': # 60%
            self.images = self.images[:int(0.6 * len(self.images))]
            self.labels = self.labels[:int(0.6 * len(self.labels))]
        elif mode == 'val': # 20% = 60% -> 80%
            self.images = self.images[int(0.6 * len(self.images)):int(0.8 * len(self.images))]
            self.labels = self.labels[int(0.6 * len(self.labels)):int(0.6 * len(self.labels))]
        else:  # 20% = 80% -> 100%
            self.images = self.images[int(0.8 * len(self.images)):]
            self.labels = self.labels[int(0.6 * len(self.labels)):]

    def load_csv(self, filename):
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in self.name2label.keys():
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*.png'))

            print(len(images), images[0])
            random.shuffle(images)
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    # image_path, 0
                    writer.writerow([img, label])
                print('writen into csv file:', filename)

        images, labels = [], []
        with open(os.path.join(self.root, filename), mode='r') as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                images.append(img)
                labels.append(int(label))
        assert len(images) == len(labels)

        return images, labels

    def __len__(self):
        return len(self.images)

    def denormalize(self, x_hat):

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        mean = torch.tensor(mean).unsqueeze(dim=1).unsqueeze(dim=1)
        std = torch.tensor(std).unsqueeze(dim=1).unsqueeze(dim=1)

        x = x_hat * std + mean
        return x

    def __getitem__(self, idx):
        # self.images, self.labels
        # idx~[0, len(images)]
        img, label = self.images[idx], self.labels[idx]

        tf = transforms.Compose([
            lambda x:Image.open(x).convert('RGB'), # string path -> image data
            transforms.Resize((int(self.resize*1.25), int(self.resize*1.25))),
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        img = tf(img)
        label = torch.tensor(label)
        return img, label


def main():
    import visdom
    viz = visdom.Visdom()
    db = Pokemon('/home/ptface02/PycharmProjects/documents/crawler/Picture/weiboPic', 224, 'train')
    # db = Pokemon(r'G:\BaiduNetdiskDownload\crawlGather\Picture\weiboPic', 224, 'train')
    sample, label = next(iter(db))
    print(sample.shape, label)
    viz.image(db.denormalize(sample), win='sample_x', opts=dict(title='sample_x'))

    loader = DataLoader(db, batch_size=32, shuffle=True) # 如果batch_size比较大，可以用多线程提取数据，num_workers=8
    # for x, y in loader:
    #     viz.images(x, nrow=8, win='batch', opts=dict(title='batch'))
    #     viz.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))

    # 如果数据格式很规整，那么可以直接采用torchvision
    # import torchvision
    # tf = transforms.Compose([
    #     transforms.Resize((64, 64)),
    #     transforms.ToTensor(),
    # ])
    # db1 = torchvision.datasets.ImageFolder(root='pokemon', transform=tf)
    # loader = DataLoader(db1, batch_size=32, shuffle=True)
    # print(db1.class_to_idx)


if __name__ == "__main__":
    main()