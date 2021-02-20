import torch.nn as nn
from GEN_config import Args

class Net(nn.Module):

    def __init__(self, args:Args):
        super(Net, self).__init__()
        self.neuralnetwork = nn.Sequential( 
            nn.Linear(args.valueStackSize*args.num_vis_pts, 4, bias = True), #stacking previous distances along with action
            nn.ReLU(),  
            nn.Linear(4, 3, bias = True), #stacking previous distances along with action
            nn.ReLU(),
            nn.Linear(3, 2, bias = True),
            nn.ReLU(),
        ) 

    def forward(self, input):
        output = self.neuralnetwork(input)
        return output