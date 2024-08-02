import requests
import os
import math
import tiktoken
import torch
from torch.utils.data import Dataset


def get_data():
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

    # write to input.txt
    input_path = os.path.join(os.getcwd(), 'Shakespeare', 'input.txt')

    # write data into file
    if not os.path.exists(input_path):
        os.mkdir(os.path.join(os.getcwd(), 'Shakespeare'))
        print("Getting data from url, please wait..........")
        response = requests.get(url).text
        print("Shakespeare data gotten!")
        # create a file
        with open(input_path, 'w', encoding='utf-8') as f:
            f.write(response)
            f.close()
        encoder = tiktoken.get_encoding('gpt2')
        data = encoder.encode_ordinary(response)
        # train : test = 8 : 2
        n = len(data)
        train, test = data[:int(n*0.8)], data[int(n*0.8):]
        train, test = torch.tensor(train), torch.tensor(test)
        torch.save(train, os.path.join(os.getcwd(), 'Shakespeare', 'train.bin'))
        torch.save(test, os.path.join(os.getcwd(), 'Shakespeare', 'test.bin'))
    train, test = torch.load(os.path.join(os.getcwd(), 'Shakespeare', 'train.bin')), torch.load(os.path.join(os.getcwd(), 'Shakespeare', 'test.bin'))
    return train, test



class MyDataset(Dataset):
    def __init__(self, data, config):
        """
        Args:
            data (list or numpy array): The data samples.
            labels (list or numpy array): The labels corresponding to the data samples.
        """
        batch_size = config.batch_size
        n = len(data)
        token_length = 512
        nums = n // token_length
        data = data[:(nums - 1) * token_length + 1]
        self.data = data[:-1].view(nums - 1, token_length)
        self.labels = data[1:].view(nums - 1, token_length)
        
        

    def __len__(self):
        # Return the total number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve the sample and label at the given index
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label
