from GEN_config import Args
from GEN_neuralnet import Net
import torch


class Agent():
    def __init__(self, args:Args, device, stateDict = None):

        self.args = args
        self.net = Net(args).double().to(device)
        self.device = device
        if stateDict != None:
            self.net.load_state_dict(stateDict)

    def chooseAction(self, state):
        state = torch.from_numpy(state).double().to(self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.net(state)
        return action.squeeze().cpu().numpy()
    
    def getParams(self):
        return self.net.state_dict()
