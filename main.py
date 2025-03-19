import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import copy
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import wandb


# 定义数据加载函数
def get_data():
    # 定义数据转换（将图像转换为Tensor并标准化）
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize((0.5,), (0.5,))  # 标准化（均值0.5，标准差0.5）
    ])

    # 下载和加载训练集和测试集
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # 使用DataLoader加载数据
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)
    return trainloader, testloader

# 定义模型
class SingleLayerNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(SingleLayerNetwork, self).__init__()
        # 定义线性层
        self.fc = nn.Linear(input_size, output_size)
        # 定义ReLU激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = (torch.tanh(x)+1)/2.0
        return x

# 定义模型
class MultiLayerNetwork(nn.Module):
    def __init__(self):
        super(MultiLayerNetwork, self).__init__()
        self.layers = nn.ModuleList()  # 用于存储逐步添加的网络层

    def add(self, layer):
        # 添加已训练好的网络层到ModuleList中
        self.layers.append(copy.deepcopy(layer))

    def pop(self):
        """删除 self.layers 中的最后一个网络层"""
        if len(self.layers) > 0:
            last_layer = self.layers[-1]  # 获取最后一层
            del self.layers[-1]  # 手动删除
            return last_layer  # 返回被删除的层
        else:
            print("Warning: No layers to remove.")
            return None

    def forward(self, x, n_layers=None, return_intermediate=False):
        outputs = []
        
        # 逐层计算输出
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if return_intermediate and (n_layers is None or i < n_layers):
                outputs.append(x)
            if i == n_layers:
                break
        
        if return_intermediate:
            return outputs
        else:
            return x
        
# 定义读出头网络
class ReadoutHead(nn.Module):
    def __init__(self, input_size, output_size):
        super(ReadoutHead, self).__init__()
        # 初始化权重为高斯分布，且权重不可训练
        self.weight = nn.Parameter(torch.randn(input_size, output_size) * 0.01, requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(output_size), requires_grad=False)

    def forward(self, x):
        # 线性变换：y = xW + b
        return torch.matmul(x, self.weight) + self.bias
    
# 定义训练流程
def train_with_readout(fixed_network, target_network, readout_head, data_loader, optimizer, criterion, device):
    if fixed_network is not None:
        fixed_network.eval()    # 固定网络不训练
    target_network.train()      # 目标网络训练
    total_loss = 0

    for inputs, labels in data_loader:
        inputs = inputs.view(inputs.shape[0], -1)  # 将图像展平
        inputs, labels = inputs.to(device), labels.to(device)

        # 如果固定网络不为空，数据先通过固定网络（不计算梯度）
        outputs = inputs
        if fixed_network is not None:
            with torch.no_grad():
                outputs = fixed_network(inputs)


        # 数据通过目标网络
        target_outputs = target_network(outputs)

        # 数据通过读出头网络
        logits = readout_head(target_outputs)

        # 计算交叉熵损失
        loss = criterion(logits, labels)
        total_loss += loss.item()

        # 反向传播优化目标网络
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / len(data_loader)

# 定义评估函数
def evaluate_accuracy(target_network, data_loader, device, readout_head=None):
    # 固定网络、目标网络和读出头网络都设置为评估模式
    target_network.eval()
    if readout_head is not None:
        readout_head.eval()

    correct = 0
    total = 0

    with torch.no_grad():  # 不计算梯度
        for inputs, labels in data_loader:
            inputs = inputs.view(inputs.shape[0], -1)  # 将图像展平
            inputs, labels = inputs.to(device), labels.to(device)

            # 数据通过目标网络
            target_outputs = target_network(inputs)

            if readout_head is not None:
                # 数据通过读出头网络
                target_outputs = readout_head(target_outputs)

            # 预测类别
            _, predicted = torch.max(target_outputs, dim=1)  # 取概率最大的类别

            # 统计正确预测的数量
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 计算并返回准确率
    accuracy = correct / total
    return accuracy

# 定义训练函数
def estimate_alpha_mle(data, x_min):
    """
    使用最大似然估计（MLE）计算幂律指数 α
    :param data: 观测数据（numpy 数组）
    :param x_min: 设定的最小阈值，幂律分布从 x_min 开始适用
    :return: 估计的 α
    """
    filtered_data = data[data >= x_min]  # 只选取大于等于 x_min 的数据
    n = len(filtered_data)  # 数据点数
    alpha = 1 + n / np.sum(np.log(filtered_data / x_min))
    return alpha

# 定义函数计算特征值
def get_eigenvalues(data):
    data = data - np.mean(data, axis=0)
    # 计算数据的协方差矩阵
    covariance_matrix = np.cov(data, rowvar=False)
    
    # 计算协方差矩阵的特征值
    eigenvalues, _ = np.linalg.eig(covariance_matrix)
    
    # 对特征值进行排序
    eigenvalues = np.sort(eigenvalues)[::-1]
    return eigenvalues

# 定义函数计算 α 和 R^2
def get_alpha_r(eigenvalues):
    # 估计幂律指数 α
    slope = estimate_alpha_mle(eigenvalues, 0.000001)

    # 计算R^2
    # 假设 eigenvalues 是特征值数组 (已按降序排列)
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)  # 计算解释方差比例
    cumulative_variance = np.cumsum(explained_variance_ratio)  # 计算累计贡献率

    # 选择前 k 个主成分，使得累计贡献率达到 95%
    k = np.argmax(cumulative_variance >= 0.95) + 1  # 找到累计方差贡献率 >= 95% 的最小维度
    R2_95 = cumulative_variance[10]
    return slope, R2_95, k

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader, testloader = get_data()

    wandb.init(
    # set the wandb project where this run will be logged
    project="FBM",
    name="LocalError",
    config={
        "dataset": "MNIST",
        "layers": 5,
        "epoch": "20",
        }
    )

    tot_NN = MultiLayerNetwork()
    input_size = 28*28
    size_range = [1000, 1000, 1000, 1000, 1000, 1000]
    for num_layer, output_size in enumerate(size_range):
        # 初始化一个单层网络
        Single_NN = SingleLayerNetwork(input_size, output_size).to(device)
        optimizer = optim.Adam(Single_NN.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        readout_head = ReadoutHead(output_size, 10).to(device)

        # 训练该单层网络
        for epoch in range(20):
            Single_NN.train()  # 设为训练模式，启用 Dropout、BatchNorm
            loss = train_with_readout(fixed_network=tot_NN, target_network=Single_NN, readout_head=readout_head, data_loader=trainloader, optimizer=optimizer, criterion=criterion, device=device)

            # 分析网络的特征
            tot_NN.add(Single_NN)
            accuracy = evaluate_accuracy(target_network=tot_NN, readout_head=readout_head, data_loader=testloader, device=device)

            tot_NN.eval()  # 设为评估模式，不启用 Dropout、BatchNorm
            output_list = []
            with torch.no_grad():  # 不计算梯度，加速推理
                for inputs, labels in trainloader:
                    inputs = inputs.view(inputs.shape[0], -1)  # 将图像展平
                    inputs, labels = inputs.to(device), labels.to(device)
                    output = tot_NN(inputs)  # 前向传播
                    output_list.append(output.cpu().numpy())  # 转换为 NumPy 并保存

            # 拼接成一个完整的 NumPy 数组
            final_output = np.vstack(output_list) 
            eigenvalues = get_eigenvalues(final_output)
            slope, R2_10, k = get_alpha_r(eigenvalues)
            tot_NN.pop()

            print("each epoch", loss, accuracy, slope, R2_10, k)
            wandb.log({f"layer{num_layer}_epoch":epoch, 
                       f"layer{num_layer}_Loss": loss, 
                       f"layer{num_layer}_accuracy":accuracy, 
                       f"layer{num_layer}_slope": slope, 
                       f"layer{num_layer}_R2_10": R2_10,
                       f"layer{num_layer}_k_95": k}) 

        wandb.log({f"layer":num_layer, 
                    f"Loss": loss, 
                    f"accuracy":accuracy, 
                    f"slope": slope, 
                    f"R2_10": R2_10,
                    f"k_95": k})
        
        input_size = output_size
        tot_NN.add(Single_NN)

        eva_value = evaluate_accuracy(target_network=tot_NN, readout_head=readout_head, data_loader=testloader, device=device)
        print("eval", eva_value)

    wandb.finish()
    tot_NN.add(readout_head)

    final_eval = evaluate_accuracy(target_network=tot_NN, data_loader=testloader, device=device)
    print(final_eval)