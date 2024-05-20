import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

lab_index = 2.1

# 设置绘图风格和字体
sns.set_theme(style="darkgrid", palette="pastel")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 创建数据
non_data = np.array([1.02,
2.12,
3.83])
attn_data = np.array([1.48,
3.11,
5.5,
8.05,
13.69,
20.91])
select_data = np.array([1.02,
2.12,
3.89,
5.88,
11.3,
18.25])
batch_size = np.array([50.48,
100.772363,
226.65,
402.86,
671.362059,
1006.971915])

improvement_percentage = ((attn_data - select_data) / attn_data) * 100


# 创建数据框
data = {
    '模型参数量': np.concatenate([batch_size[:len(non_data)], batch_size, batch_size]),
    '计算时间': np.concatenate([non_data, attn_data, select_data]),
    '重计算方法': ['Non'] * len(non_data) + ['Attn'] * len(attn_data) + ['Select'] * len(select_data)
}

df = pd.DataFrame(data)

# 绘制柱状图
plt.figure(figsize=(12, 8))
sns.barplot(x='模型参数量', y='计算时间', hue='重计算方法', data=df, palette='pastel')

# 添加提高的百分比标签
for i, (batch, improvement) in enumerate(zip(batch_size, improvement_percentage)):
    plt.text(x=i, y=(attn_data[i] + select_data[i]) / 2, s=f'{improvement:.2f}%', ha='center', va='baseline')

# 设置标题和标签
plt.title('不同模型参数量大小下的计算时间')
plt.xlabel('模型参数量/M')
plt.ylabel('平均计算时间')

plt.grid(which='major', linestyle='--', linewidth='0.3', color='gray')
plt.grid(which='minor', linestyle='-.', linewidth='0.25', color='gray')

# 显示图例
plt.legend(title='重计算方法')

# 显示图像
plt.savefig(f'/Users/mac/Desktop/SelectiveRecompute/data/pic/实验{lab_index}.jpeg', dpi=500, bbox_inches='tight')
plt.show()
