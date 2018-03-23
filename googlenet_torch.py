import torch
import torch.legacy
import torch.legacy.nn
import torch.nn as nn
from torch.autograd import Variable
from functools import reduce

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func,self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func,self.forward_prepare(input))


googlenet = nn.Sequential( # Sequential,
	nn.Conv2d(3,64,(7, 7),(2, 2),(3, 3)),
	nn.ReLU(),
	nn.MaxPool2d((3, 3),(2, 2),(0, 0),ceil_mode=True),
	Lambda(lambda x,lrn=torch.legacy.nn.SpatialCrossMapLRN(*(5, 0.0001, 0.75, 1)): Variable(lrn.forward(x.data))),
	nn.Conv2d(64,64,(1, 1)),
	nn.ReLU(),
	nn.Conv2d(64,192,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	Lambda(lambda x,lrn=torch.legacy.nn.SpatialCrossMapLRN(*(5, 0.0001, 0.75, 1)): Variable(lrn.forward(x.data))),
	nn.MaxPool2d((3, 3),(2, 2),(0, 0),ceil_mode=True),
	nn.Conv2d(192,64,(1, 1)),
	nn.ReLU(),
	nn.Conv2d(192,96,(1, 1)),
	nn.ReLU(),
	nn.Conv2d(96,128,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(192,16,(1, 1)),
	nn.ReLU(),
	nn.Conv2d(16,32,(5, 5),(1, 1),(2, 2)),
	nn.ReLU(),
	nn.MaxPool2d((3, 3),(1, 1),(1, 1),ceil_mode=True),
	nn.Conv2d(192,32,(1, 1)),
	nn.ReLU(),
	nn.Conv2d(256,128,(1, 1)),
	nn.ReLU(),
	nn.Conv2d(256,128,(1, 1)),
	nn.ReLU(),
	nn.Conv2d(128,192,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(256,32,(1, 1)),
	nn.ReLU(),
	nn.Conv2d(32,96,(5, 5),(1, 1),(2, 2)),
	nn.ReLU(),
	nn.MaxPool2d((3, 3),(1, 1),(1, 1),ceil_mode=True),
	nn.Conv2d(256,64,(1, 1)),
	nn.ReLU(),
	nn.MaxPool2d((3, 3),(2, 2),(0, 0),ceil_mode=True),
	nn.Conv2d(480,192,(1, 1)),
	nn.ReLU(),
	nn.Conv2d(480,96,(1, 1)),
	nn.ReLU(),
	nn.Conv2d(96,208,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(480,16,(1, 1)),
	nn.ReLU(),
	nn.Conv2d(16,48,(5, 5),(1, 1),(2, 2)),
	nn.ReLU(),
	nn.MaxPool2d((3, 3),(1, 1),(1, 1),ceil_mode=True),
	nn.Conv2d(480,64,(1, 1)),
	nn.ReLU(),
	nn.AvgPool2d((5, 5),(3, 3),(0, 0),ceil_mode=True),#AvgPool2d,
	nn.Conv2d(512,128,(1, 1)),
	nn.ReLU(),
	Lambda(lambda x: x.view(x.size(0),-1)), # View,
	nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(2048,1024)), # Linear,
	nn.ReLU(),
	nn.Dropout(0.7),
	nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(1024,365)), # Linear,
)
