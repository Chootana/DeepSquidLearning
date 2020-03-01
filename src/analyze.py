import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import preprocess 


def load_all_results():
    data_dir = './data/processed'
    win = np.load('{}/win.npy'.format(data_dir))

    general_keys = [
        'stage',
        'rule',
        'start_time',
        'elapsed_time',
        'x_power',
        'my_team_count',
        'other_team_count',
    ]

    general = {}
    for key in general_keys:
        general[key] = preprocess.load_results(data_dir, 'general', key)

    battle_keys = [
        'kill_count',
        'death_count',
        'assist_count',
        'special_count',
        'game_paint_point',
    ]

    alpha = beta = {}
    for key in battle_keys:
        alpha[key] = preprocess.load_results(data_dir, 'alpha', key)
        beta[key] = preprocess.load_results(data_dir, 'beta', key)

    
    return general, alpha, beta, win

def make_diff_table(results, key):
    key_alpha = 'alpha_{}'.format(key)
    key_beta = 'beta_{}'.format(key)
    diff = results[key_alpha] - results[key_beta]
    
    diff_win = diff[results['win']==1]
    diff_win_positive = diff_win[diff_win>0]
    diff_win_negative = diff_win[diff_win<0]
    diff_win_equal = diff_win[diff_win==0]

    diff_lose = diff[results['win']==0]
    diff_lose_positive = diff_lose[diff_lose>0]
    diff_lose_negative = diff_lose[diff_lose<0]
    diff_lose_equal = diff_lose[diff_lose==0]

    print('--------------------------------')

    print('-{}-\t\t[win]\t[lose]'.format(key))
    print('[positive]\t{:.1f}\t{:.1f}'.format(
        len(diff_win_positive)/len(diff_win)*100,
        len(diff_lose_positive)/len(diff_lose)*100
        )
    )

    print('[negative]\t{:.1f}\t{:.1f}'.format(
        len(diff_win_negative)/len(diff_win)*100,
        len(diff_lose_negative)/len(diff_lose)*100
        )
    )

    print('[equal]\t\t{:.1f}\t{:.1f}'.format(
        len(diff_win_equal)/len(diff_win)*100,
        len(diff_lose_equal)/len(diff_lose)*100
        )
    )

    print('--------------------------------')


def show_stage_win_rate(dgeneral):

    stage_keys = dgeneral['stage'].unique()
    rule_keys = dgeneral['rule'].unique()
    for stage_key in stage_keys:
        for rule_key in rule_keys:
            print(dgeneral[(dgeneral.stage == stage_key) & (dgeneral.rule == rule_key) & (dgeneral.win == True)])

def make_hist(results, key):
    key_alpha = 'alpha_{}'.format(key)
    
    data = results[key_alpha]
    data_win = data[results['win']==1]
    data_lose = data[results['win']==0]

    # plt.hist(data_win, bins=10, color='red')
    plt.hist(data_win, bins=10, color='blue')
    plt.savefig('hist_win_{}.png'.format(key))

if __name__ == "__main__":
    general, alpha, beta, win = load_all_results()

    dgeneral = pd.DataFrame(general)
    dgeneral['win'] = win

    show_stage_win_rate(dgeneral)


    # 各ルールごとの各ステージの勝率
    # プレイ時間の勝率


    # make_diff_table(results, 'kill')
    # make_diff_table(results, 'death')
    # make_diff_table(results, 'assist')
    # make_diff_table(results, 'special')
    # make_diff_table(results, 'paint') 

    # make_hist(results, 'death')
