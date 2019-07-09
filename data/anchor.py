import os
import torch
import numpy as np
from lxml import etree
from torch.utils.data.dataset import Dataset


class AnchorDataset(Dataset):
	def __init__(self, root):
		self.__root = root
		self.__files = self.__find_files()
		self.__boxes = self.__load_boxes()
		self.__len = self.__boxes.shape[0]

	def __find_files(self):
		files = list(map(lambda x: os.path.join(self.__root, x), os.listdir(self.__root)))
		image_files = list(filter(lambda x: '.xml' not in x, files))
		annotation_files = list(filter(lambda x: '.xml' in x, files))

		return [
			annotation_file for annotation_file in annotation_files
							for image_file in image_files
				if os.path.splitext(annotation_file)[0] == os.path.splitext(image_file)[0]
		]

	def __load_boxes(self):
		boxes = []

		for file in self.__files:
			tree = etree.parse(file)
			root = tree.getroot()

			for obj in root.findall('object'):
				bndbox = obj.find('bndbox')
				x = int(bndbox.find('xmin').text)
				y = int(bndbox.find('ymin').text)
				w = int(bndbox.find('xmax').text) - x
				h = int(bndbox.find('ymax').text) - y

				boxes.append([w, h])
		return np.array(boxes)

	def numpy(self):
		return self.__boxes.copy()

	def __len__(self):
		return self.__len

	def __getitem__(self, idx):
		return torch.from_numpy(self.__boxes[idx])
