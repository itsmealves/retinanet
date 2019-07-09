import torch
from model.network import RetinaNet
from data.anchor import AnchorDataset


dataset = AnchorDataset('/home/gabriel/Documents/HELMINTOS-TT/train')

x = torch.zeros(4, 3, 224, 224)
model = RetinaNet(n_input=3, n_classes=3, n_anchors=9, anchor_set=dataset)
model(x)
