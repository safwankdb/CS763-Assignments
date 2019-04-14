import torch

device = 'cpu'


class Data():

    def __init__(self, test=False, actualTest=False, m_train=950, m_test=234, D=154, data='data/train_data.txt', label='data/train_labels.txt'):
        """ Tokenize first """
        self.actualTest=actualTest
        train_data_file = "data/train_data.txt"
        test_data_file = "data/test_data.txt"
        values = []
        with open(train_data_file, "r") as f:
            for line in f:
                values.append(line)
        # print(len(values))
        with open(test_data_file, "r") as f:
            for line in f:
                values.append(line)
        combined = ' '.join(values)
        values = combined.split(' ')
        values = list(set(values))  # removing duplicate elements
        if('\n' in values):
            values.remove('\n')
        values.sort()
        self.mapping = {}
        for i in range(0, len(values)):
            self.mapping[int(values[i])] = i+1

        """ Prepare Y """
        if(not self.actualTest):
            self.Y = torch.zeros(m_train+m_test)
            with open(label, "r") as f:
                cnt = 0
                for line in f:
                    self.Y[cnt] = int(line[:-1])
                    cnt += 1

        """ Prepare X in raw form, i.e. not one hot"""
        self.X = []
        with open(data, "r") as f:
            for line in f:
                values = line.split(' ')
                values = values[:-1]
                self.X.append(values)

        self.D = D

        if(not self.actualTest):
            if test:
                self.m = m_test
                print('Validation Mode')
                self.X = self.X[m_train:]
                self.Y = self.Y.narrow(0, m_train, m_test)
                if(self.Y.shape[0] != len(self.X)):
                    print("Wrong dimensions in validation")

            else:
                self.m = m_train
                print('Train Mode')
                self.X = self.X[:m_train]
                self.Y = self.Y.narrow(0, 0, m_train)
                if(self.Y.shape[0] != len(self.X)):
                    print("Wrong dimensions in training")
        else:
            self.m=len(self.X)

    def sample(self, batch_size, index):
        if index + batch_size >= self.m:
            index = self.m - batch_size
        examples_X = []
        examples_y = []
        maxLen = 0
        for i in range(index, index + batch_size):
            if(len(self.X[i]) > maxLen):
                maxLen = len(self.X[i])

        for i in range(index, index + batch_size):
            temp = torch.zeros(maxLen, self.D)
            for k in range(0, len(self.X[i])):
                temp[k][self.mapping[int(self.X[i][k])]] = 1
            for k in range(len(self.X[i]), maxLen):
                temp[k][0] = 1
            if(not self.actualTest):
                examples_y.append(self.Y[i])
            examples_X.append(temp)
            if(not self.actualTest):
                batch = torch.stack(examples_X).double().to(\
                    device), torch.stack(examples_y).double().to(device)
            else:
                batch = torch.stack(examples_X).double().to(device)

        return batch
