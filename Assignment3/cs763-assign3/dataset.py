import torch.utils.data
import torchfile


class Data(torch.utils.data.Dataset):
    """Data in data.bin"""

    def __init__(self, test=False):
        device = 'cuda:0'
        m_train = 22400
        if test:
            self.isTest = True
            print('Test')
            self.X = torchfile.load('data.bin')[m_train:]
            self.y = torchfile.load('labels.bin')[m_train:]
            self.X = torch.from_numpy(self.X).double().view(self.X.shape[0], -1).to(device)
        else:
            self.isTest = False
            self.device = 'cuda:0'
            self.X = torchfile.load('data.bin')[:m_train]
            self.y = torchfile.load('labels.bin')[:m_train]
            self.X = torch.from_numpy(self.X).double().view(self.X.shape[0], -1).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index] / 255., self.y[index]
