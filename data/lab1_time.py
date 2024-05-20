import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

lab_index = 1.1

# 设置绘图风格和字体
sns.set_theme(style="darkgrid", palette="pastel")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 创建数据
non_data = np.array([1.88, 2.33, 2.18])
attn_data = np.array([2.7, 3.4, 3.12, 3.55, 4.33, 5.29, 7.86, 10.43, 13.22])
select_data = np.array([1.85, 2.48, 2.18, 2.73, 3.33, 4.34, 6.7, 9.17, 11.76])
batch_size = np.array([20, 25, 22, 27, 32, 40, 60, 80, 100])

improvement_percentage = ((attn_data - select_data) / attn_data) * 100


# 创建数据框
data = {
    'batch-size': np.concatenate([batch_size[:len(non_data)], batch_size, batch_size]),
    '计算时间': np.concatenate([non_data, attn_data, select_data]),
    '重计算方法': ['Non'] * len(non_data) + ['Attn'] * len(attn_data) + ['Select'] * len(select_data)
}

df = pd.DataFrame(data)

# 绘制柱状图
plt.figure(figsize=(12, 8))
sns.barplot(x='batch-size', y='计算时间', hue='重计算方法', data=df, palette='pastel')

# 添加提高的百分比标签
for i, (batch, improvement) in enumerate(zip(batch_size, improvement_percentage)):
    plt.text(x=i, y=(attn_data[i] + select_data[i]) / 2, s=f'{improvement:.2f}%', ha='center', va='baseline')

# 设置标题和标签
plt.title('不同批量大小下的计算时间')
plt.xlabel('批量大小')
plt.ylabel('平均计算时间')

plt.grid(which='major', linestyle='--', linewidth='0.3', color='gray')
plt.grid(which='minor', linestyle='-.', linewidth='0.25', color='gray')

# 显示图例
plt.legend(title='重计算方法')

# 显示图像
plt.savefig(f'/Users/mac/Desktop/SelectiveRecompute/data/pic/实验{lab_index}.jpeg', dpi=500, bbox_inches='tight')
plt.show()
