import torch
from torch import nn

class ClassificationModel(nn.Module):

	def __init__(self,in_features,out_features,hidden_units):
		super().__init__()
		self.layer_1 = nn.Linear(in_features=in_features,out_features=hidden_units)
		self.layer_2 = nn.Linear(in_features=hidden_units, out_features=hidden_units)
		self.layer_3 = nn.Linear(in_features=hidden_units, out_features= out_features)
		self.relu = nn.ReLU()

	def forward(self,X:torch.Tensor):
		z = self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(X)))))
		return z