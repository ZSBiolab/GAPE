#去除尾缀
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from age_pre_model import AgeModel  # 从model.py导入模型
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import matplotlib.pyplot as plt

# 检查CUDA是否可用，然后使用GPU或回退到CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 加载数据
df_train = pd.read_csv('data/train_data_demo.csv')
df_pre = pd.read_csv('data/train_data_demo.csv')

# 定义路径变量
results_path = 'results'

selected_data_train = df_train.iloc[1:, 1:]  # 假设第一行为标签行，第一列不是特征
selected_data_pre = df_pre.iloc[1:, 1:]  # 假设第一行为标签行，第一列不是特征

# 提取标签和数据
labels_train = df_train.iloc[0, 1:]  # 第一行作为标签
labels_pre = df_pre.iloc[0, 1:]  # 第一行作为标签
data_train = selected_data_train.T  # 转置使每一行成为一个样本
data_pre = selected_data_pre.T  # 转置使每一行成为一个样本

# 编码标签
label_encoder = LabelEncoder()
encoded_labels_train = label_encoder.fit_transform(labels_train)
print(f"总共有 {len(set(encoded_labels_train))} 种标签。")

# 数据转换为PyTorch张量
X = torch.tensor(data_pre.values.astype(float)).float().to(device)

# 创建数据加载器
dataset = TensorDataset(X)
data_loader = DataLoader(dataset, batch_size=8)
input_embedding = 512

# 加载模型
model_path = os.path.join(results_path, 'run/Age_model.pt')
model = AgeModel(X.shape[1], input_embedding, len(set(encoded_labels_train))).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# 预测
all_outputs = []
all_embedding = []
with torch.no_grad():
    for inputs, in data_loader:
        inputs = inputs.to(device)
        embedding, outputs = model(inputs)
        all_outputs.extend(outputs.cpu().numpy())
        all_embedding.extend(embedding.cpu().numpy())

# 检查 all_embedding 和 labels_pre 的形状是否匹配
all_embedding = np.array(all_embedding).T  # 现在形状是 (input_embedding, 样本数)
if all_embedding.shape[1] != len(labels_pre):
    raise ValueError(f"Shape mismatch: all_embedding has {all_embedding.shape[1]} samples, but labels_pre has {len(labels_pre)} labels")

# 格式化输出为DataFrame，并确保列名唯一
def make_unique(labels):
    seen = {}
    result = []
    for label in labels:
        if label not in seen:
            seen[label] = 1
            result.append(label)
        else:
            seen[label] += 1
            result.append(f"{label}.{seen[label]}")
    return result

unique_labels_pre = make_unique(labels_pre.astype(str))  # 确保所有标签都是字符串
output_df = pd.DataFrame(all_embedding, columns=unique_labels_pre)

# 去除列名中的后缀
def remove_suffix(label):
    parts = label.split('.')
    return parts[0]

clean_labels = [remove_suffix(str(label)) for label in output_df.columns]  # 确保列名是字符串

output_df.columns = clean_labels
output_df.to_csv(os.path.join(results_path, 'run/output_clinic_data_embedding.csv'), index=False)
print("Outputs are saved to 'results_aus_A_maxmin_Our/run/output_train_data_embedding.csv'.")

# 转换输出为标签
predicted_labels = label_encoder.inverse_transform(np.argmax(all_outputs, axis=1))
# 从原始标签解码真实标签
original_labels_pre = label_encoder.inverse_transform(label_encoder.transform(labels_pre))
# 创建DataFrame并保存到CSV
df_predictions = pd.DataFrame({
    'True Label': original_labels_pre,
    'Predicted Label': predicted_labels
})

# 标记预测错误为False，正确为True
df_predictions['Mismatch'] = (df_predictions['True Label'] != df_predictions['Predicted Label']).astype(bool)
df_predictions['Mismatch'] = ~df_predictions['Mismatch']  # 取反，使错误为False，正确为True

# 计算预测准确率
accuracy = df_predictions['Mismatch'].mean()
print(f"预测准确率: {accuracy:.2%}")
df_predictions.to_csv(os.path.join(results_path, 'run/pre_age.csv'), index=False)
print("Predictions are saved to 'results_aus_A_maxmin_Our/run/pre_age.csv'.")

# 重新加载生成的预年龄文件并进行可视化
df_result = pd.read_csv(os.path.join(results_path, 'run/pre_age.csv'))

# 可视化
plt.figure(figsize=(12, 8))
plt.grid(True, linestyle='--', alpha=0.7)

# 设置标记点的透明度
plt.scatter(df_result.index, df_result['True Label'], color='blue', marker='^', label='Real Age', alpha=0.6, edgecolors='w', s=100)
plt.scatter(df_result.index, df_result['Predicted Label'], color='red', marker='o', label='Predicted Age', alpha=0.6, edgecolors='w', s=100)

# 标记相同值
for i in range(len(df_result)):
    if df_result['True Label'][i] == df_result['Predicted Label'][i]:
        plt.annotate('Match', (df_result.index[i], df_result['True Label'][i]),
                     textcoords="offset points", xytext=(0, -15), ha='center', color='green', fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", edgecolor="green", facecolor="white"),
                     arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=.2", color='green'))

plt.xlabel('Sample Index')
plt.ylabel('Age')
plt.legend(loc='upper right')
plt.title('Real Age vs Predicted Age', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
