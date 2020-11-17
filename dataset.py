import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class CUHK_PEDES(Dataset):
    """ the class for CUHK_PEDES dataset

    Attributes:

    """

    def __init__(self, conf, dataset, is_train=False, image_caption='image'):
        """ init CUHK_PEDES class

        Args:
            conf:
            dataset:
        """
        self.split = dataset[0]["split"]
        self.image_caption = image_caption
        self.dataset = dataset
        self.config = conf
        self.positive_samples = conf.positive_samples
        self.negative_samples = conf.negative_samples
        self.n_original_captions = conf.n_original_captions
        self.transform = transforms.Compose([
            transforms.Resize(conf.image_size),
            transforms.ToTensor()
        ])
        conf.logger.info(f'init {self.split} {image_caption} ,length:{len(self)}, dataset length:{len(self.dataset)}')
        if self.image_caption == 'caption':
            self.num_classes = len(self) / self.n_original_captions
        elif self.image_caption == 'image':
            self.num_classes = len(self)

    def __getitem__(self, index):
        """ get an item of dataset by index

        Args:
            index: the index of the dataset

        Returns:
            item: an item of dataset
        """
        if self.image_caption == 'caption':
            img_index = index // self.n_original_captions
        elif self.image_caption == 'image':
            img_index = index
        data = self.dataset[img_index]
        image_path = os.path.join(self.config.images_dir, data['file_path'])
        cap_index = index % self.n_original_captions
        caption_indexes = data['index_captions'][cap_index]
        p_id = int(data['id'])
        image = Image.open(image_path)
        # resize image to 256x256
        image = self.transform(image)
        # caption
        caption = np.zeros(self.config.max_length)
        for i, word_i in enumerate(caption_indexes):
            if i < self.config.max_length:
                caption[i] = word_i
        caption = torch.LongTensor(caption)
        return index, image, caption, img_index, p_id

    def __len__(self):
        """get the length of the dataset

        Returns:
            the length of the dataset
        """
        if self.image_caption == 'caption':
            return len(self.dataset) * self.n_original_captions
        elif self.image_caption == 'image':
            return len(self.dataset)
