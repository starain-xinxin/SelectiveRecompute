import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import pandas as pd

lab_index = 1

def plot_BER(BER, SNR, encode_method, decode_method, save_dir, figsize=(10, 8), dpi=500):
        sns.set_theme(style="darkgrid", palette="pastel")
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        min_snr = min(SNR)
        max_snr = max(SNR)
        interval = math.floor((max_snr - min_snr) / len(SNR) * 2) / 2
        i = 0
        j = 0
        for en_method in encode_method:
            plt.figure(figsize=figsize)
            for de_method in decode_method:
                data = {'SNR': SNR, 'BER': BER[i * len(encode_method) + j]}
                df = pd.DataFrame(data)
                sns.lineplot(x='SNR', y='BER', data=df, label=f'{de_method}', linewidth=2.5)
                j = j + 1
            plt.yscale('log')
            plt.title(f'{en_method}编码方案下，各种解码方案的误比特率')
            plt.grid(which='major', linestyle='--', linewidth='0.3', color='gray')
            plt.grid(which='minor', linestyle='-.', linewidth='0.25', color='gray')
            plt.xticks(np.arange(min_snr, max_snr, interval))  # 从0到60，间隔为5
            plt.savefig(save_dir + f'/BER-SNR:{en_method}编码方案.jpeg', dpi=dpi, bbox_inches='tight')
            plt.show()
            i = i + 1

# def my_plot(figsize=(10, 8), dpi=500):
#     sns.set_theme(style="darkgrid", palette="pastel")
#     plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
#     plt.rcParams['axes.unicode_minus'] = False
#     plt.figure(figsize=figsize)

#     data = {'batch-size': , '计算速度': }
#     df = pd.DataFrame(data)
#     sns.lineplot(x='batch-size', y='计算速度', data=df, label=f'', linewidth=2.5)

#     plt.yscale('log')
#     plt.title(f'')
#     plt.grid(which='major', linestyle='--', linewidth='0.3', color='gray')
#     plt.grid(which='minor', linestyle='-.', linewidth='0.25', color='gray')
#     # plt.xticks(np.arange(min_snr, max_snr, interval))  # 从0到60，间隔为5
#     plt.savefig(f'./pic/实验{lab_index}.jpeg', dpi=dpi, bbox_inches='tight')
#     plt.show()

#lab1
sns.set_theme(style="darkgrid", palette="pastel")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(10, 8))

non_data = np.array([1.88, 2.33, 2.18])
attn_data = np.array([2.7, 3.4, 3.12, 3.55, 4.33, 5.29, 7.86, 10.43, 13.22])
select_data = np.array([1.85, 2.48, 2.18, 2.73, 3.33, 4.34, 6.7, 9.17, 11.76])
batch_size = np.array([20, 25, 22, 27, 32, 40, 60, 80, 100])

data0 = {'batch-size': batch_size[:3], '计算速度': non_data}
data1 = {'batch-size': batch_size, '计算速度': attn_data}
data2 = {'batch-size': batch_size, '计算速度': select_data}

df0 = pd.DataFrame(data0)
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

sns.lineplot(x='batch-size', y='计算速度', data=df0, label=f'', linewidth=2.5)
sns.lineplot(x='batch-size', y='计算速度', data=df1, label=f'', linewidth=2.5)
sns.lineplot(x='batch-size', y='计算速度', data=df2, label=f'', linewidth=2.5)

# plt.title(f'')
plt.grid(which='major', linestyle='--', linewidth='0.3', color='gray')
plt.grid(which='minor', linestyle='-.', linewidth='0.25', color='gray')
# plt.xticks(np.arange(min_snr, max_snr, interval))  # 从0到60，间隔为5
plt.savefig(f'/Users/mac/Desktop/SelectiveRecompute/data/pic/实验{lab_index}.jpeg', dpi=500, bbox_inches='tight')
plt.show()

