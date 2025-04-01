import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from rna_pre_model import prernamodel  # 从model.py导入模型
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np
# 检查CUDA是否可用，然后使用GPU或回退到CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 加载数据
df = pd.read_csv('data/train_data_demo.csv')
df_initial_model = pd.read_csv('data/train_data_demo.csv')
df_embedding = pd.read_csv('results/run/output_clinic_data_embedding.csv')

df_label_selected_data = df.iloc[1:, 1:]
df_input_selected_data = df.iloc[1:, 1:]

pre_age_labels_initial_model = df_initial_model.iloc[0, 1:]  # 第一行作为标签
pre_age_labels = df.iloc[0, 1:]  # 第一行作为标签
pre_age_label_encoder = LabelEncoder()
encoded_pre_age_labels_initial_model = pre_age_label_encoder.fit_transform(pre_age_labels_initial_model)

encoded_pre_age_labels = pre_age_label_encoder.transform(pre_age_labels)

print(f"总共有 {len(set(encoded_pre_age_labels_initial_model))} 种标签。")
pre_age_labels = torch.tensor(encoded_pre_age_labels).long().to(device)
print(f'pre_age_labels shape: {pre_age_labels.shape}')

# 提取输入和标签数据
inputs = df_label_selected_data.values.astype(float)
embedding = df_embedding.values.astype(float)
labels = df_label_selected_data.values.astype(float)

# 数据转换为PyTorch张量
inputs = torch.tensor(inputs).float().t().to(device)
labels = torch.tensor(labels).float().t().to(device)
embedding = torch.tensor(embedding).float().t().to(device)

print("inputs shape:", inputs.shape)
print("labels shape:", labels.shape)
print("embedding shape:", embedding.shape)
embedding_size = 512

# 创建数据加载器
dataset = TensorDataset(inputs, labels, embedding, pre_age_labels)
data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# 创建模型实例并加载预训练权重
pre_rna_model = prernamodel(inputs.shape[1], labels.shape[1], embedding_size, len(set(encoded_pre_age_labels_initial_model))).to(device)

# 加载预训练权重
pretrained_weights_path = 'results/run/RNA_model.pt'
pretrained_dict = torch.load(pretrained_weights_path, map_location=device)
model_dict = pre_rna_model.state_dict()

# 更新模型的state_dict
pre_rna_model.load_state_dict(pretrained_dict)

# 将模型设置为评估模式
pre_rna_model.eval()

# 创建用于存储结果的列表
pre_rna_outputs1_list = []

# 禁用梯度计算
with torch.no_grad():
    for inputs, labels, embedding, pre_age_labels in data_loader:
        inputs, labels, embedding, pre_age_labels = inputs.to(device), labels.to(device), embedding.to(device), pre_age_labels.to(device)
        mask = (torch.rand(inputs.shape) > 0.05).float().to(device)
        pre_rna_inputs_masked = inputs * mask
        pre_rna_outputs1, _ = pre_rna_model(pre_rna_inputs_masked, embedding)
        pre_rna_outputs1_list.append(pre_rna_outputs1.cpu().numpy())

# 将结果转换为DataFrame并保存到CSV文件
pre_rna_outputs1_array = np.vstack(pre_rna_outputs1_list)


output_matrix = pre_rna_outputs1_array.transpose()
new_df = pd.read_csv('data/train_data_demo.csv')
new_df.iloc[1:, 1:] = output_matrix
new_df.to_csv('results/output.csv', index=False)
print("result already save to results/output.csv")
