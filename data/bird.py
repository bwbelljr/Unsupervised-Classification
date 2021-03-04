from torch.utils.data import Dataset
import PIL
import pandas as pd
from PIL import Image
from torchvision import transforms as tf
from fastai.data.transforms import get_image_files
from fastai.data.external import untar_data,URLs
from fastai.vision.all import *
import re
from utils.mypath import MyPath

class Birds(Dataset):
      def __init__(self, root=MyPath.db_root_dir('bird'), split='train', transform=None):
        super(Birds, self).__init__()

        self.split = split
        
        self.transform = transform

        self.resize = tf.Resize(256)


        path = untar_data(URLs.CUB_200_2011)


        self.files = get_image_files(path/"images")
        self.label = dict(sorted(enumerate(set(self.files.map(self.label_func))), key=itemgetter(1)))
        self.labels = dict([(value, key) for key, value in self.label.items()])
        self.df = pd.read_csv(path/'train_test_split.txt',delimiter=' ')


        if self.split == 'train':
          self.file_index = [i['1'] for i in self.df.to_dict('records') if i['0']==1]
          self.Files= [i for i in self.files if self.splitter(i) in self.file_index]


        else:
          self.file_index = [i['1'] for i in self.df.to_dict('records') if i['0']==0]
          self.Files = [i for i in self.files if self.splitter(i) in self.file_index]




      def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        Img= self.Files[index]
        target = self.labels[self.label_func(Img)]

        # make consistent with all other datasets
        # return a PIL Image

        with open(str(Img), 'rb') as f:
          img = Image.open(f).convert('RGB')


        img_size = img.size
        img = self.resize(img)


        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'target': target, 'meta': {'im_size': img_size, 'index': index}}

        return out


      def label_func(self, fname):
        return re.match(r'^(.*)_\d+_\d+.jpg$', fname.name).groups()[0]

      def splitter(self, fname):
        return int(re.match(r'\w+_(.*)_\d+.jpg$', fname.name).groups()[0])

      def get_image(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self.resize(img)
        return img
