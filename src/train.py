import numpy as np 
import matplotlib.pyplot as plt 
import glob
import dataloader 
import os 
import torch 
import torch.optim as optim
import torch.nn as nn  
from torch.utils.data import DataLoader, TensorDataset

from model import Net


def train():

    # config
    epochs = 100
    dim_input = 10 # 

    # model setting
    train_loader, test_loader = dataloader.get_dataloader()
    model = Net(dim_input)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


    print('start training...')
    train_loss_list = []
    test_loss_list = []
    for epoch in range(epochs):

        # train
        model.train()
        train_loss = []
        train_pred = []
        for data in train_loader:
            input = data['input']
            target = data['target']

            output = model(input)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

            # prediction
            correct = torch.max(output.detach(), 1)[1] == target
            train_pred.append(sum(correct.numpy())/len(correct.numpy()))

        train_loss_list.append(sum(train_loss) / len(train_loss))

        # test
        model.eval()
        test_loss = []
        test_pred = []
        for data in test_loader:
            input = data['input']
            target = data['target']

            output = model(input)
            loss = criterion(output, target)

            test_loss.append(loss.item())

            # prediction
            correct = torch.max(output.detach(), 1)[1] == target
            test_pred.append(sum(correct.numpy())/len(correct.numpy()))

        test_loss_list.append(sum(train_loss) / len(train_loss))

        # log 
        print('[{}/{}] loss(train): {:.3f} pred(train): {:.3f} \t loss(test): {:.3f} pred(test): {:.3f}'.format(
            epoch+1, epochs, 
            sum(train_loss) / len(train_loss),
            sum(train_pred) / len(train_pred),
            sum(test_loss) / len(test_loss),
            sum(test_pred) / len(test_pred),
        ))

    save_results(model, train_loss_list, test_loss_list)


def save_results(model, train_loss_list, test_loss_list):

    save_path = './result'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_loss_plot(save_path, train_loss_list, test_loss_list)
    save_model_weight(save_path, model)


def save_loss_plot(save_path, train_loss_list, test_loss_list):
    """save loss plot
    
    Parameters
    ----------
    save_path : str
    train_loss_list : list
    test_loss_list : list
    """

    plt.plot(train_loss_list)
    plt.plot(test_loss_list)
    plt.savefig('{}/loss.png'.format(save_path))


def save_model_weight(save_path, model):
    """save model weight
    
    Parameters
    ----------
    save_path : str
    model : Net
    """
    weight_path = '{}/weight.pth'.format(save_path)
    torch.save(model.state_dict(), weight_path)
    

if __name__ == '__main__':
    train()