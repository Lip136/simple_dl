# encoding:utf-8
import os, csv, glob, random
import torch
from torch.utils.data import Dataset
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

        if mode=='train':


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

    def __getitem__(self, idx):
        # self.images, self.labels
        # idx~[0, len(images)]
        img, label = self.images[idx], self.labels[idx]

        tf = transforms.Compose([
            lambda x:Image.open(x).convert('RGB'), # string path -> image data
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor()
        ])

        img = tf(img)
        label = torch.tensor(label)
        return img, label

def main():
    db = Pokemon('/home/ptface02/PycharmProjects/documents/crawler/Picture/weiboPic', 224, 'train')
if __name__ == "__main__":
    main()