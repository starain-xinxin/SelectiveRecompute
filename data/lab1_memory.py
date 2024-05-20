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
non_data = np.array([58.374781,
72.87011,
64.17733]) / 80 * 100
attn_data = np.array([13.68722,
17.01031,
15.0172377,
18.338863,
21.661464,
26.978,
40.268,
55.55844,
66.8848]) / 80 * 100
select_data = np.array([57.6,
62.32237,
57.5563,
70.546,
65.1615,
61.665,
70.268,
76.68,
67.63]) / 80 * 100
batch_size = np.array([20, 25, 22, 27, 32, 40, 60, 80, 100])

improvement_percentage = ((- attn_data + select_data) / attn_data) * 100


# 创建数据框
data = {
    'batch-size': np.concatenate([batch_size[:len(non_data)], batch_size, batch_size]),
    '显存效率': np.concatenate([non_data, attn_data, select_data]),
    '重计算方法': ['Non'] * len(non_data) + ['Attn'] * len(attn_data) + ['Select'] * len(select_data)
}

df = pd.DataFrame(data)

# 绘制柱状图
plt.figure(figsize=(12, 8))
sns.barplot(x='batch-size', y='显存效率', hue='重计算方法', data=df, palette='pastel')

# 添加提高的百分比标签
for i, (batch, improvement) in enumerate(zip(batch_size, improvement_percentage)):
    plt.text(x=i, y=(attn_data[i] + select_data[i]) / 2, s=f'{improvement:.2f}%', ha='center', va='baseline')

# 设置标题和标签
plt.title('不同批量大小下的显存效率')
plt.xlabel('批量大小')
plt.ylabel('显存效率')

plt.grid(which='major', linestyle='--', linewidth='0.3', color='gray')
plt.grid(which='minor', linestyle='-.', linewidth='0.25', color='gray')

# 显示图例
plt.legend(title='重计算方法')

# 显示图像
plt.savefig(f'/Users/mac/Desktop/SelectiveRecompute/data/pic/实验{lab_index}memory.jpeg', dpi=500, bbox_inches='tight')
plt.show()
