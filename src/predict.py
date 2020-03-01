import numpy as np 
import torch 
import torch.nn as nn 
from model import Net
from dataloader import get_dataloader


def predict(model):
    criterion = nn.CrossEntropyLoss() 

    _, test_loader = get_dataloader()

    pred_list = []
    output_list = []
    target_list = []
    for data in test_loader:
        input = data['input']
        target = data['target']

        output = model(input)
        
        correct = torch.max(output.detach(), 1)[1] == target
        pred_list.append(sum(correct.numpy()) / len(correct.numpy()))
        output_list.append(output.detach().numpy())
        target_list.append(target.detach().numpy())
    
    print('Correct answer rate: {:.1f} %'.format(
        sum(pred_list) / len(pred_list) * 100
    ))

if __name__ == '__main__':
    data_dir = './result'
    weight_path = '{}/weight.pth'.format(data_dir)

    dim_input = 10
    model = Net(dim_input)
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    print(model)

    predict(model)