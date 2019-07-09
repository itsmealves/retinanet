import torch
from torch import nn
from model.fpn import FPN
from sklearn.cluster import KMeans


class RetinaNet(nn.Module):
	def __init__(self, n_input, n_classes, anchors=None, n_anchors=9, anchor_set=None, c=256):
		super().__init__()
		
		self.__c = c
		self.__n_input = n_input
		self.__n_classes = n_classes
		
		self.fpn = FPN(c)
		self.anchors = anchors

		if anchors is None:
			self.anchors = self.__find_anchors(n_anchors, anchor_set)
		self.__a = len(self.anchors)

		print(self.anchors)

		self.box_subnets = nn.Sequential(*self.__get_subnets(4 * self.__a))
		self.class_subnets = nn.Sequential(*self.__get_subnets(n_classes * self.__a))

	def __find_anchors(self, n_anchors, anchor_set):
		kmeans = KMeans(n_anchors)
		kmeans.fit(anchor_set.numpy())
		return kmeans.cluster_centers_

	def __get_subnets(self, output):
		subnets = []
		for n in range(self.fpn.n):
			subnet = nn.Sequential(
				nn.Conv2d(self.__c, self.__c, 3, padding=1),
				nn.ReLU(),
				nn.Conv2d(self.__c, self.__c, 3, padding=1),
				nn.ReLU(),
				nn.Conv2d(self.__c, self.__c, 3, padding=1),
				nn.ReLU(),
				nn.Conv2d(self.__c, self.__c, 3, padding=1),
				nn.ReLU(),
				nn.Conv2d(self.__c, output, 3, padding=1),
				nn.Sigmoid()
			)

			subnets.append(subnet)
		return subnets

	def forward(self, x):
		box_predictions = []
		class_predictions = []

		for idx, scale in self.fpn(x):
			box_predictions.append(self.box_subnets[idx](scale))
			class_predictions.append(self.class_subnets[idx](scale))
			print(box_predictions[-1].size(), class_predictions[-1].size())
		return x
