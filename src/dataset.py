import os 
import numpy as np 
import torch
from torch.utils.data import Dataset 
from analyze import load_all_results

class SquidDataset(Dataset):
    def __init__(self):
        super(SquidDataset, self).__init__()

        self.general, self.alpha, self.beta, self.win = load_all_results()
        self.items = np.vstack(
            [
                self.alpha['kill_count'],
                self.beta['kill_count'],
                self.alpha['death_count'],
                self.beta['death_count'],
                self.alpha['assist_count'],
                self.beta['assist_count'],
                self.alpha['special_count'],
                self.beta['special_count'],
                self.alpha['game_paint_point'],
                self.beta['game_paint_point'],
            ]
        ).transpose()

        self.mean, self.std = self.load_mean_std()

    def load_mean_std(self):
        """load mean & std for standardization
        
        Returns
        -------
        mean : ndarray
        std : ndarray
        """
        data_path = './data/processed/mean_std'
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        mean_path = '{}/mean.npy'.format(data_path)
        std_path = '{}/std.npy'.format(data_path)

        if os.path.isfile(mean_path) and os.path.isfile(std_path):
            print('load mean.npy: {}'.format(mean_path))
            mean = np.load(mean_path)

            print('load std.npy: {}'.format(std_path))
            std = np.load(std_path)
        else:
            mean, std = self.generate_mean_std(mean_path, std_path)
        
        return mean, std
    
    def generate_mean_std(self, mean_path, std_path):
        """calculation mean & std for standardaztion
        
        Parameters
        ----------
        mean_path : str
        std_path : str
            
        Returns
        -------
        mean : ndarray
        std : ndarray
        """
        print('generate mean.npy: {}'.format(mean_path))
        mean = np.mean(self.items, axis=0)
        np.save(mean_path, mean)

        print('generate std.npy: {}'.format(std_path))
        std = np.std(self.items, axis=0)
        # from IPython import embed; embed(); exit()
        std[std==0] = 1e-9
        np.save(std_path, std)

        return mean, std

    def input_preprocessing(self, data):
        """input  preprocessing 
        
        Parameters
        ----------
        data : ndarray  
            input data
        
        Returns
        -------
        data : torch.tensor
            preprocessed input data
        """

        # standardization
        data = (data - self.mean) / self.std

        data = torch.tensor(data, dtype=torch.float32)
        return data
    
    def target_preprocessing(self, data):
        """target preprocessing
        
        Parameters
        ----------
        data : ndarray
            target data
        
        Returns
        -------
        data : torch.tensor
            preprocessed target data
        """

        data = torch.tensor(data)
        return data
    
    def __getitem__(self, index):
        input = self.items[index]
        target = self.win[index]
        
        input = self.input_preprocessing(input)
        target = self.target_preprocessing(target)
        return {
            'input': input,
            'target': target,
        }
        
    def __len__(self):
        return len(self.win)

if __name__ == '__main__':
    # check
    squid_dataset = SquidDataset('train')
