import torchfile
import torch

device = 'cpu'

class Data():
    
    def __init__(self, test=False, m_train=950, m_test=250, data='data/train_data.txt', label='data/train_labels.txt'):

        """ Tokenize first """
        test_data_file="data/test_data.txt"
        values=[]
        with open(data, "r") as f:
            for line in f:
                values.append(line)
        with open(test_data_file, "r") as f:
            for line in f:
                values.append(line)
        combined=' '.join(values)
        values=combined.split(' ')
        values=list(set(values)) # removing duplicate elements
        print ("Number of unique tokens = ",len(values))
        self.mapping={}
        for i in range(0, len(values)):
            self.mapping[int(values[i])]=i+1

        """ Prepare Y """
        self.Y=torch.zeros(m_train+m_test, 2)
        with open(label,"r") as f:
            cnt=0
            for line in f:
                self.Y[cnt][int(line)]=1

        """ Prepare X in raw form, i.e. not one hot"""
        self.X=[]
        with open(data, "r") as f:
            for line in f:
                values=' '.split(line)
                self.X.append(values)

        self.D=154

        if test:
            self.m = m_test
            print('Test Mode')
            self.X=self.X[m_train:]
            self.Y=self.Y.narrow(0,m_train,m_test)
            if(self.Y.shape[0]!=len(self.X)):
                print ("Wrong dimensions in testing")

        else:
            self.m = m_train
            print('Train Mode')
            self.X=self.X[:m_train]
            self.Y=self.Y.narrow(0,0,m_train)
            if(self.Y.shape[0]!=len(self.X)):
                print ("Wrong dimensions in training")        

    def sample(self, batch_size, index):
        if index + batch_size > self.m:
            index = self.m - batch_size
        examples_X = []
        examples_y = []
        maxLen=0
        for i in range(index, index + batch_size):
            if(len(self.X[i])>maxLen):
                maxLen=len(self.X[i])

        for i in range(index, index + batch_size):
            temp=torch.zeros(maxLen, self.D)
            for k in range(0, len(self.X[i])):
                temp[k][self.X[i][k]]=1
            for k in range(len(self.X[i], maxLen)):
                temp[k][0]=1
            examples_y.append(self.Y[i])
            examples_X.append(temp)
            batch = torch.stack(examples_X), torch.stack(examples_y)
        return batch
