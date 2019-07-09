from torch import nn
from torchvision import models
import torch.nn.functional as F


class FPN(nn.Module):
	def __init__(self, c, pretrained=False):
		super().__init__()
		base_model = models.resnet101(pretrained=pretrained)

		self.layer0 = nn.Sequential(
			base_model.conv1,
			base_model.bn1,
			base_model.relu,
			base_model.maxpool
		)

		self.layer1 = base_model.layer1
		self.layer2 = base_model.layer2
		self.layer3 = base_model.layer3
		self.layer4 = base_model.layer4

		self.maxpool = nn.MaxPool2d(1, stride=2)
		self.toplayer = nn.Conv2d(2048, c, kernel_size=1, stride=1, padding=0)
		self.latlayer1 = nn.Conv2d(1024, c, kernel_size=1, stride=1, padding=0)
		self.latlayer2 = nn.Conv2d(512, c, kernel_size=1, stride=1, padding=0)
		self.latlayer3 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
		self.smooth1 = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1)
		self.smooth2 = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1)
		self.smooth3 = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1)

	def forward(self, x):
		c1 = self.layer0(x)
		c2 = self.layer1(c1)
		c3 = self.layer2(c2)
		c4 = self.layer3(c3)
		c5 = self.layer4(c4)

		p5 = self.toplayer(c5)
		p4 = self.__upsample(p5, self.latlayer1(c4), self.smooth1)
		p3 = self.__upsample(p4, self.latlayer2(c3), self.smooth2)
		p2 = self.__upsample(p3, self.latlayer3(c2), self.smooth3)

		p6 = self.maxpool(p5)
		return enumerate([p2, p3, p4, p5, p6])

	@property
	def n(self):
		return 5

	def __upsample(self, feature_map, lateral_map, smooth_layer):
		_, _, h, w = lateral_map.size()
		output = F.interpolate(feature_map, size=(h, w), mode='bicubic', align_corners=False)
		return smooth_layer(output + lateral_map)


