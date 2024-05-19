import torch
from torch.utils.data import DataLoader, Dataset


class DataMaker(Dataset):
    """ 随机数据类 """

    def __init__(self, batch_size=10, seq_len=1024, d_model=512, num_class=20, iters=5):
        rand_data = torch.randn(iters*batch_size, seq_len, d_model)
        rand_labels = torch.randint(low=0, high=num_class, size=(iters*batch_size,))
        self.data = rand_data
        self.labels = rand_labels
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_class = num_class

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # label = torch.randint(low=0, high=self.num_class, size=())
        label = self.labels[idx]
        return sample, label

# # 示例数据
# data = torch.randn(100, 3, 32, 32)  # 100个32x32的彩色图像
# labels = torch.randint(0, 10, (100,))  # 100个标签
#
# # 创建数据集和DataLoader
# dataset = CustomDataset(data, labels, transform=transform)
# dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=2)
#
# # 训练循环示例
# model = torch.nn.Linear(3 * 32 * 32, 10).cuda()
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# num_epochs = 5
#
# for epoch in range(num_epochs):
#     for batch_idx, (inputs, targets) in enumerate(dataloader):
#         inputs, targets = inputs.cuda(), targets.cuda()
#
#         # 训练代码
#         optimizer.zero_grad()
#         outputs = model(inputs.view(inputs.size(0), -1))  # 扁平化输入
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()
#
#         if batch_idx % 10 == 0:
#             print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
