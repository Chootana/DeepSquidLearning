import pandas as pd
import json 
import glob 
import os 
from tqdm import tqdm 
import numpy as np 
import matplotlib.pyplot as plt 


class GameResult():
    """parse game result
    """

    def __init__(self, data_path):
        
        self.team_keys = [
            'player_result',
            'my_team_members',
            'other_team_members',
        ]

        self.result_keys = [
            'kill_count',
            'assist_count',
            'death_count',
            'special_count',
            'game_paint_point',
            'sort_score',
        ]

        self.game = self.load_data(data_path)
    
        try:
            self.general = {
                'stage': self.game['stage'][0]['name'],
                'rule': self.game['rule'][0]['name'],
                'start_time': self.game['start_time'][0],
                'my_team_count': self.game['my_team_count'][0],
                'elapsed_time': self.game['elapsed_time'][0],
                'x_power': self.game['x_power'][0],
                'other_team_count': self.game['other_team_count'][0],
            }

            self.alpha = [
                self.game[self.team_keys[0]][0],
                self.game[self.team_keys[1]][0][0],
                self.game[self.team_keys[1]][0][1],
                self.game[self.team_keys[1]][0][2],
            ]
            
            self.beta = [
                self.game[self.team_keys[2]][0][0],
                self.game[self.team_keys[2]][0][1],
                self.game[self.team_keys[2]][0][2],
                self.game[self.team_keys[2]][0][3],
            ]
        except Exception as e: 
            self.general = {
                'error': e,
            }
            self.alpha = {
                'error': e,
            }
            self.beta = {
                'error': e,
            }


    def load_data(self, data_path):
        """load data
        
        Parameters
        ----------
        data_path : str
            data path
        
        Returns
        -------
        df_dict : dict
            data frame dict
        """

        with open(data_path, 'r') as f:
            js = '[' + ','.join(f) + ']'
        
        df = pd.read_json(js)
        df_dict = df.to_dict() 
        self.check_keys(df_dict, self.team_keys)

        return df_dict

    def check_keys(self, data_dict, keys_list):
        """check whether there are kyes we want in data
        
        Parameters
        ----------
        data_dict : dict    
        keys_list : list
        """

        for key_name in keys_list:
            if key_name not in data_dict.keys():
                assert False, '{} key is not in data'.format(key_name)
        
        return 

    def load_all_players_results(self):
        """load all player's results
        
        Returns
        -------
        win : dict
        alpha : dict
        beta : dict
            4 VS 4
        """
        alpha = {}
        beta = {}

        for key in self.result_keys:
            if key not in self.alpha:
                alpha[key] = 0
            for num in range(len(self.alpha)):
                alpha[key] += self.alpha[num][key]

            if key not in self.beta:
                beta[key] = 0
            for num in range(len(self.beta)):
                beta[key] += self.beta[num][key]

        if self.game['my_team_result'][0]['name'] == 'WIN!':
            win = 1
        else:
            win = 0

        return win, alpha, beta

    def load_general_results(self):
        return self.general


def get_one_game_result(one_game, player):
    """get one game's result
    
    Parameters
    ----------
    one_game : GameResult
        
    player : dict
        
    
    Returns
    -------
    results : list
        
    """ 
    results = one_game.load_results(player)
    return results


def get_data_path_list(data_dir):
    """get data path 
    
    Parameters
    ----------
    data_dir : str
        
    Returns
    -------
    data_path_list : list
        data/####.json, ...,
    """
    dirs = os.listdir(data_dir)

    data_path_list = []
    for dir_json in dirs:
        if '.' in dir_json or 'processed' in dir_json:
            continue
        else:
            data_path_list.extend(glob.glob('{}/{}/*.json'.format(data_dir, dir_json)))
    
    return data_path_list


def process():
    data_dir = './data'
    data_path_list = get_data_path_list(data_dir)

    battle_results = []
    win_list = []
    alpha_dict = {}
    beta_dict = {}
    general_dict = {}
    
    for data_path in tqdm(data_path_list):
        # if one_game has something problems, the result is skipped
        try:
            one_game = GameResult(data_path)

            if one_game.game['type'][0] == 'league': 
                continue # only gachi

            general = one_game.load_general_results()

            if 'error' in general.keys():
                continue # exclude it because of S+ battle, ...

            for key in general.keys():
                if key not in general_dict:
                    general_dict[key] = []
                general_dict[key].append(general[key])


            win, alpha, beta = one_game.load_all_players_results()
            win_list.append(win)

            for key in alpha.keys():
                if key not in alpha_dict:
                    alpha_dict[key] = []
                alpha_dict[key].append(alpha[key])
            
            for key in beta.keys():
                if key not in beta_dict:
                    beta_dict[key] = []
                beta_dict[key].append(beta[key])

        except IndexError:
            continue
    
    return win_list, alpha_dict, beta_dict, general_dict


def load_results(data_dir, name, key):
    """
    Parameters
    ----------
    data_dir : str
    name : str
        alpha, beta, general
    key : str
        dict keys
    
    Returns
    -------
    results : ndarray
    """

    # Do you check it is safe?
    results = np.load('{}/{}_{}.npy'.format(data_dir, name, key), allow_pickle=True)
    return results


def plot_result(data_dir, win, alpha, beta, key):
    """
    
    Parameters
    ----------
    data_dir : str
    win : ndarray 
    alpha : dict
    beta : dict
    key : str
        dict keys
    """

    save_dir = '{}/figure'.format(data_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    win = np.array(win)
    alpha_value = np.array(alpha[key])
    beta_value = np.array(beta[key])
    
    plt.scatter(beta_value[win==True], alpha_value[win==True], color='red')
    plt.scatter(beta_value[win==False], alpha_value[win==False],
    color='blue')
    plt.title(key)
    plt.xlabel('beta')
    plt.ylabel('alpha')
    plt.savefig('{}/{}.png'.format(save_dir, key))


def save_npy(data_dir, dict, name, key):
    """save npy-file
    
    Parameters
    ----------
    data_dir : str
    dict : dict
        values
    name : str
        alpha, beta, ganeral
    key : str
        dict keys
    """
    dict_value = np.array(dict[key])
    np.save('{}/{}_{}'.format(data_dir, name, key), dict_value)


if __name__ == '__main__':

    win, alpha, beta, general = process()

    data_dir = './data/processed'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    np.save('{}/win.npy'.format(data_dir), win)

    key_battle_list = [
        'kill_count',
        'assist_count',
        'death_count',
        'special_count',
        'game_paint_point',
    ]

    for key in key_battle_list:
        plot_result(data_dir, win, alpha, beta, key)
        save_npy(data_dir, alpha, 'alpha', key)
        save_npy(data_dir, beta, 'beta', key)
    
    key_general_list = [
        'stage',
        'rule',
        'start_time',
        'my_team_count',
        'other_team_count',
        'elapsed_time',
        'x_power',
    ]

    for key in key_general_list:
        save_npy(data_dir, general, 'general', key)