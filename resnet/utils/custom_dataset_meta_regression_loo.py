import torch
import os, glob
import random, csv

# import visdom
from torch.utils.data import Dataset, DataLoader
# import torchvision
from torchvision import transforms
from PIL import Image


class Carbon(Dataset):

    def __init__(self, root, resize, mode, subclassname):
        super(Carbon, self).__init__()

        self.root = root
        self.resize = resize
        self.soil_carbon = [1.3456, 2.2976, 3.8787, 3.1541]
        self.sand_carbon = [1.54, 3.03, 4.25, 4.90, 14.14, 18.97, 31.58]

        self.name2label = {}
        # num_folders = len(sorted(os.listdir(os.path.join(root))));
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):
                continue
            if root[0:4] == 'soil':
                self.name2label[name] = self.soil_carbon[len(self.name2label.keys())]
            else:
                self.name2label[name] = self.sand_carbon[len(self.name2label.keys())]

                

        print(self.name2label)

        self.moisture2label = {}
        
        # image, label
        if mode == 'train': 
            self.images, self.wifi_images, self.labels, self.metas = self.load_csv('train_images.csv', subclassname)
        elif mode == 'val':
            self.images, self.wifi_images, self.labels, self.metas = self.load_csv('val_images.csv', subclassname)
        elif mode == 'all':
            self.images, self.wifi_images, self.labels, self.metas = self.load_csv('train_images.csv', subclassname)
            self.images2, self.wifi_images2, self.labels2, self.metas2 = self.load_csv('val_images.csv', subclassname)
            self.images = self.images + self.images2
            self.wifi_images = self.wifi_images + self.wifi_images2
            self.labels = self.labels + self.labels2
            self.metas = self.metas + self.metas2


    def load_csv(self, filename, subclass):
        if not os.path.exists(os.path.join(self.root, filename)):
            images, wifi_images = [], []
            for name in self.name2label.keys():
                images += glob.glob(os.path.join(self.root, name, 'img', '*.png'))
                images += sorted(glob.glob(os.path.join(self.root, name, 'img', '*.jpg')))
                images += glob.glob(os.path.join(self.root, name, 'img', '*.jpeg'))
                wifi_images += glob.glob(os.path.join(self.root, name, subclass, '*.png'))
                wifi_images += sorted(glob.glob(os.path.join(self.root, name, subclass, '*.jpg')))
                wifi_images += glob.glob(os.path.join(self.root, name, subclass, '*.jpeg'))

            print(len(images), images)
            print(len(wifi_images), wifi_images)

            # load meta data
            meta_data_all = []
            with open(os.path.join(self.root, 'soil_data.csv')) as f:
                reader = csv.reader(f, delimiter=' ')
                for row in reader:
                    row_new = [a for a in row if a != '']
                    e = float(row_new[0])
                    oc = float(row_new[1])
                    vwc = float(row_new[2])
                    meta_data_all.append([e, oc, vwc])
            print(len(meta_data_all))

            # for img in images:
            for i in range(len(images)):
                img = images[i]
                img_name = img.split(os.sep)[-1]                
                moisture = img_name[0:5]
                if moisture not in self.moisture2label.keys():
                    self.moisture2label[moisture] = len(self.moisture2label.keys())
            print(len(self.moisture2label.keys()))
            print(self.moisture2label)
        
            moisture_idxs = list(range(len(self.moisture2label.keys())))
            random.shuffle(moisture_idxs)
                        # valid_moisture_idxs = []
            # if len(moisture_idxs) == 24 or len(moisture_idxs) == 16:
            #     for i in range(4):
            #         valid_moisture_idxs.append(i * 4 + random.randint(0, 3))
            #     train_moisture_idxs = [i for i in moisture_idxs if i not in valid_moisture_idxs]
            # else:
            #     train_moisture_idxs = moisture_idxs[:int(0.75 * len(moisture_idxs))]
            valid_moisture_idxs = [i for i in moisture_idxs]
            train_moisture_idxs = []
            print(sorted(train_moisture_idxs))
            print(sorted(valid_moisture_idxs))
            train_images, train_wifi_images, train_labels = [], [], []
            val_images, val_wifi_images, val_labels = [], [], []
            train_meta, val_meta = [], []
            for i in range(len(images)):
                img = images[i]              
                name = img.split(os.sep)[-3]
                wifi_img = wifi_images[i]
                meta = meta_data_all[i]
                assert img.split(os.sep)[-1] == wifi_img.split(os.sep)[-1]
                label = self.name2label[name]
                img_name = img.split(os.sep)[-1]                
                moisture = img_name[0:5]
                moisture_label = self.moisture2label[moisture]
                if moisture_label in train_moisture_idxs:
                    train_images.append(img)           
                    train_labels.append(label)
                    train_wifi_images.append(wifi_img)
                    train_meta.append(meta)
                else:
                    val_images.append(img)                   
                    val_labels.append(label)
                    val_wifi_images.append(wifi_img)
                    val_meta.append(meta)

            assert len(train_images) == len(train_labels)

            
            # train_lists = list(zip(train_images, train_wifi_images))
            # random.shuffle(train_lists)
            # train_images, train_wifi_images = zip(*train_lists)
            random_index = [x for x in range(len(train_images))]
            random.shuffle(random_index)
            train_images = [train_images[a] for a in random_index]
            train_wifi_images = [train_wifi_images[a] for a in random_index]
            train_meta = [train_meta[a] for a in random_index]


            # val_lists = list(zip(val_images,  val_wifi_images))
            # random.shuffle(val_lists)
            # val_images, val_wifi_images = zip(*val_lists)
            random_index = [x for x in range(len(val_images))]
            random.shuffle(random_index)
            val_images = [val_images[a] for a in random_index]
            val_wifi_images = [val_wifi_images[a] for a in random_index]
            val_meta = [val_meta[a] for a in random_index]


            with open(os.path.join(self.root, 'train_images.csv'), mode='w', newline='') as f:
                writer = csv.writer(f)
                # for img in images:
                for i in range(len(train_images)):
                    img = train_images[i]
                    wifi_img = train_wifi_images[i]
                    meta = train_meta[i]
                    name = img.split(os.sep)[-3]
                    label = self.name2label[name]
                    # assert label == train_labels[i]
                    # 'pokemon\\bulbasaur\\00000000.png', 0
                    writer.writerow([img, wifi_img, label, meta[0], meta[1], meta[2]])
                print(f"written into csv file: {'train_images.csv'}")

            with open(os.path.join(self.root, 'val_images.csv'), mode='w', newline='') as f:
                writer = csv.writer(f)
                # for img in images:
                for i in range(len(val_images)):
                    img = val_images[i]
                    wifi_img = val_wifi_images[i]
                    meta = val_meta[i]
                    name = img.split(os.sep)[-3]                    
                    label = self.name2label[name]
                    # assert label == val_labels[i]
                    # 'pokemon\\bulbasaur\\00000000.png', 0
                    writer.writerow([img, wifi_img, label, meta[0], meta[1], meta[2]])
                print(f"written into csv file: {'val_images.csv'}")

        # read from csv file
        images, wifi_images, labels = [], [], []
        metas = []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                # 'pokemon\\bulbasaur\\00000000.png', 0
                img, wifi_img, label, meta1, meta2, meta3 = row                
                meta = [float(meta1)/100.0, float(meta2)/100.0, float(meta3)/100.0]
                label = float(label) - float(meta2)
                
                images.append(img)
                wifi_images.append(wifi_img)
                labels.append(label)
                metas.append(meta)

        assert len(images) == len(labels)




        return images, wifi_images, labels, metas

    def __len__(self):
        return len(self.images)

    def denormalize(self, x_hat):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # x_hat = (x-mean)/std
        # x = x_hat * std + mean
        # x: [c, h, w]
        # mean: [3] => [3, 1, 1] for broadcast to happen
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)

        x = x_hat * std + mean

        return x

    def __getitem__(self, idx):
        # idx: [0:len(images)]
        # self.images, self.labels
        # img: 'pokemon\\bulbasaur\\00000000.png'
        # label: 0
        img, wifi_img, label = self.images[idx], self.wifi_images[idx], self.labels[idx]
        meta = self.metas[idx]

        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),  # str path -> img data
            transforms.CenterCrop(self.resize),
            transforms.ToTensor()
        ])

        img = tf(img)
        wifi_img = tf(wifi_img)
        label = torch.tensor(label)
        meta = torch.tensor(meta)

        return img, wifi_img, label, meta


def main():
    import visdom
    import time

    # viz = visdom.Visdom(use_incoming_socket=False)
    
    # using self implemented Dataset class
    db = Carbon('soil_combined', 224, 'train')
    print(db.images[0], db.wifi_images[0], db.labels[0])

    x, y, z = next(iter(db))
    print(x.shape, y.shape, z)

    # viz.image(db.denormalize(x), win='sample_x', opts=dict(title='sample_x'))

    loader = DataLoader(db, batch_size=32, shuffle=True)

    # for x, y in loader:
    #     viz.images(db.denormalize(x), nrow=8, win='batch', opts=dict(title='batch'))
    #     viz.text(str(y.numpy()), win='label', opts=dict(title='batch_y'))

    #     time.sleep(10)


    # # using torchvision.datasets.ImageFolder
    # tf = transforms.Compose([
    #     transforms.Resize((64, 64)),
    #     transforms.ToTensor(),
    #     ])
    #
    # db = torchvision.datasets.ImageFolder(root='pokemon', transform=tf)
    # loader = DataLoader(db, batch_size=32, shuffle=True, num_workers=)
    #
    # print(db.class_to_idx)
    #
    # for x, y in loader:
    #     viz.images(x, nrow=8, win='batch', opts=dict(title='batch'))
    #     viz.text(str(y.numpy()), win='label', opts=dict(title='batch_y'))
    #
    #     time.sleep(10)

if __name__ == '__main__':
    main()