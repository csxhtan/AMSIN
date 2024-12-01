import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import csv
import glob
import random
import cv2


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, crop_size=(128, 128), mode='train'):
        super(type(self), self).__init__()
        self.blurry_img_list = []
        self.clear_img_list = []
        self.snow_img_list = []
        self.crop_size = crop_size
        self.to_tensor = transforms.ToTensor()
        self.root = root
        self.nameclass = {}

        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):
                continue
            self.nameclass[name] = len(self.nameclass.keys())
        print(self.nameclass)
        self.blurry_img_list, self.clear_img_list = self.load_csv('data.csv')

        print(len(self.blurry_img_list))

    def load_csv(self, filename):
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in self.nameclass.keys():
                # 'all\\gt\\00001.jpg
                images += glob.glob(os.path.join(self.root, name, '*.png'))

            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:  # 'all\\gt\\00000000.jpg'
                    name = img.split(os.sep)[-2]
                    category = self.nameclass[name]
                    # 'all\\gt\\00000000.png', 0
                    writer.writerow([img, category])
                print('writen into csv file:', filename)

        blurry_img, clear_img = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                # 'all\\gt\\00000000.jpg', 0
                img, category = row
                category = int(category)
                if category == 1:
                    clear_img.append(img)
                elif category == 0:
                    blurry_img.append(img)

        return blurry_img, clear_img

    def crop_resize_totensor(self, img, crop_location):
        img256 = img.crop(crop_location)
        img128 = img256.resize((self.crop_size[0] // 2, self.crop_size[1] // 2), resample=Image.BILINEAR)
        img64 = img128.resize((self.crop_size[0] // 4, self.crop_size[1] // 4), resample=Image.BILINEAR)
        return self.to_tensor(img256), self.to_tensor(img128), self.to_tensor(img64)

    def __len__(self):
        return len(self.clear_img_list)

    def __getitem__(self, idx):
        # filename processing
        blurry_img_name = self.blurry_img_list[idx]
        clear_img_name = self.clear_img_list[idx]

        blurry_left_img = Image.open(blurry_img_name)
        clear_img = Image.open(clear_img_name)

        w = blurry_left_img.size[0]
        h = blurry_left_img.size[1]
        self.crop_size = (w, h)

        crop_left = int(np.floor(np.random.uniform(0, blurry_left_img.size[0] - self.crop_size[0] + 1)))
        crop_top = int(np.floor(np.random.uniform(0, blurry_left_img.size[1] - self.crop_size[1] + 1)))
        crop_location = (crop_left, crop_top, crop_left + self.crop_size[0], crop_top + self.crop_size[1])

        img256_left, img128_left, img64_left = self.crop_resize_totensor(blurry_left_img, crop_location)
        label256, label128, label64 = self.crop_resize_totensor(clear_img, crop_location)
        batch = {'img256': img256_left, 'img128': img128_left,
                 'img64': img64_left, 'label256': label256, 'label128': label128,
                 'label64': label64}
        # for k in batch:
        #     batch[k] = batch[k] * 2 - 1.0  # in range [-1,1]
        return batch


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, root, crop_size=(128, 128)):
        super(type(self), self).__init__()
        self.blurry_img_list = []
        self.clear_img_list = []
        self.snow_img_list = []
        self.crop_size = crop_size
        self.to_tensor = transforms.ToTensor()
        self.root = root

        self.nameclass = {}

        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):
                continue
            self.nameclass[name] = len(self.nameclass.keys())
        print(self.nameclass)
        self.blurry_img_list, self.clear_img_list = self.load_csv('data.csv')

        # self.blurry_img_list = self.blurry_img_list[int(0.802 * len(self.blurry_img_list)):]
        # self.clear_img_list = self.clear_img_list[int(0.802 * len(self.clear_img_list)):]
        print(len(self.blurry_img_list))

    def load_csv(self, filename):
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in self.nameclass.keys():
                # 'all\\gt\\00001.jpg
                images += glob.glob(os.path.join(self.root, name, '*.png'))

            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:  # 'all\\gt\\00000000.jpg'
                    name = img.split(os.sep)[-2]
                    category = self.nameclass[name]
                    # 'all\\gt\\00000000.png', 0
                    writer.writerow([img, category])
                print('writen into csv file:', filename)

        blurry_img, clear_img, = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                # 'all\\gt\\00000000.jpg', 0
                img, category = row
                category = int(category)
                if category == 1:
                    clear_img.append(img)
                elif category == 0:
                    blurry_img.append(img)

        return blurry_img, clear_img

    def crop_resize_totensor(self, img, crop_location):
        img256 = img.crop(crop_location)
        img128 = img256.resize((self.crop_size[0] // 2, self.crop_size[1] // 2), resample=Image.BILINEAR)
        img64 = img128.resize((self.crop_size[0] // 4, self.crop_size[1] // 4), resample=Image.BILINEAR)
        return self.to_tensor(img256), self.to_tensor(img128), self.to_tensor(img64)

    def __len__(self):
        return len(self.clear_img_list)

    def __getitem__(self, idx):
        # filename processing
        blurry_img_name = self.blurry_img_list[idx]
        clear_img_name = self.clear_img_list[idx]

        blurry_left_img = Image.open(blurry_img_name)
        clear_img = Image.open(clear_img_name)

        w = blurry_left_img.size[0]
        h = blurry_left_img.size[1]

        w = w - (int(w) % 8)
        h = h - (int(h) % 8)
        self.crop_size = (w, h)

        crop_left = int(np.floor(np.random.uniform(0, blurry_left_img.size[0] - self.crop_size[0] + 1)))
        crop_top = int(np.floor(np.random.uniform(0, blurry_left_img.size[1] - self.crop_size[1] + 1)))
        crop_location = (crop_left, crop_top, crop_left + self.crop_size[0], crop_top + self.crop_size[1])

        img256_left, img128_left, img64_left = self.crop_resize_totensor(blurry_left_img, crop_location)
        label256, label128, label64 = self.crop_resize_totensor(clear_img, crop_location)
        batch = {'img256': img256_left, 'img128': img128_left,
                 'img64': img64_left, 'label256': label256, 'label128': label128,
                 'label64': label64}
        # for k in batch:
        #     batch[k] = batch[k] * 2 - 1.0  # in range [-1,1]
        return batch


class TestDataset_all(torch.utils.data.Dataset):
    def __init__(self, root):
        super(type(self), self).__init__()
        self.blurry_img_list = []
        self.clear_img_list = []
        self.snow_img_list = []
        self.to_tensor = transforms.ToTensor()
        self.root = root

        self.nameclass = {}

        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):
                continue
            self.nameclass[name] = len(self.nameclass.keys())
        print(self.nameclass)
        self.blurry_img_list, self.clear_img_list, self.snow_img_list = self.load_csv('test_data_S.csv')

        print(len(self.blurry_img_list))

    def load_csv(self, filename):
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in self.nameclass.keys():
                # 'all\\gt\\00001.jpg
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))

            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:  # 'all\\gt\\00000000.jpg'
                    name = img.split(os.sep)[-2]
                    category = self.nameclass[name]
                    # 'all\\gt\\00000000.png', 0
                    writer.writerow([img, category])
                print('writen into csv file:', filename)

        blurry_img, clear_img, snow_img = [], [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                # 'all\\gt\\00000000.jpg', 0
                img, category = row
                category = int(category)
                if category == 0:
                    clear_img.append(img)
                elif category == 1:
                    snow_img.append(img)
                else:
                    blurry_img.append(img)

        assert len(clear_img) == len(snow_img) and len(snow_img) == len(blurry_img)

        return blurry_img, clear_img, snow_img

    def resize_totensor(self, img):
        img_size = img.size
        img256 = img
        img128 = img256.resize((img_size[0] // 2, img_size[1] // 2), resample=Image.BILINEAR)
        img64 = img128.resize((img_size[0] // 4, img_size[1] // 4), resample=Image.BILINEAR)
        return self.to_tensor(img256), self.to_tensor(img128), self.to_tensor(img64)

    def __len__(self):
        return len(self.clear_img_list)

    def __getitem__(self, idx):
        # filename processing
        blurry_img_name = self.blurry_img_list[idx]
        clear_img_name = self.clear_img_list[idx]
        snow_mask_name = self.snow_img_list[idx]

        blurry_img = Image.open(blurry_img_name)
        clear_img = Image.open(clear_img_name)
        snow_mask = Image.open(snow_mask_name)
        assert blurry_img.size == clear_img.size and clear_img.size == snow_mask.size

        img256, img128, img64 = self.resize_totensor(blurry_img)
        label256, label128, label64 = self.resize_totensor(clear_img)
        mask256, mask128, mask64 = self.resize_totensor(snow_mask)
        batch = {'img256': torch.cat([img256, torch.zeros_like(img256)], dim=0), 'img128': img128,
                 'img64': img64, 'label256': label256, 'label128': label128,
                 'label64': label64, 'mask256': mask256, 'mask128': mask128, 'mask64': mask64}
        for k in batch:
            batch[k] = batch[k] * 2 - 1.0  # in range [-1,1]
        return batch


if __name__ == '__main__':
    # imgs = []
    # h = 10000
    # w = 10000
    # with open(os.path.join('data/Train', 'data.csv')) as f:
    #     reader = csv.reader(f)
    #     for row in reader:
    #         # 'all\\gt\\00000000.jpg', 0
    #         img, category = row
    #         category = int(category)
    #         if category == 0:
    #             imgs.append(img)
    # print(len(imgs))
    # for image in imgs:
    #     i = Image.open(image)
    #     if i.size[0] < h:
    #         h = i.size[0]
    #     if i.size[1] < w:
    #         w = i.size[1]
    #
    # print(h, w)
    # image = cv2.imread('data/Train/Gt/1.tif', -1)
    # cv2.namedWindow('img')
    # cv2.moveWindow('img', 400, 400)
    # cv2.imshow('img', image)
    # cv2.waitKey(1000000)
    # to_tensor = transforms.ToTensor()
    # img = to_tensor(image)
    # img = torch.flip(img, dims=[0])
    # print(img.shape)
    # img = img.permute(2, 1, 0)
    # print(img.shape)
    # m = transforms.ToPILImage()(img)
    # m.show()
    # print(m.size)
    # image = Image.open('data/Train/Gt/1.tif')
    # image.show()
    # img = np.array(image)
    # print(img.shape)
    # t = torch.from_numpy(img)
    # print(t.shape)
    # img = to_tensor(image)
    # print(img.shape)
    dataset = Dataset('data/UIEBD', mode='train')
