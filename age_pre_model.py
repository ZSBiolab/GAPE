import torch
import torch.nn as nn
import torch.nn.functional as F

class AgeModel(nn.Module):
    def __init__(self, output_features, input_embedding, num_classes):
        super(AgeModel, self).__init__()
        # 这些层的维度需要与原始模型中对应层的维度一致
        self.attention = SelfAttention(output_features)
        self.residual_block3 = ResidualBlock(output_features, 1024)
        self.residual_block4 = ResidualBlock(1024, input_embedding)
        self.fc2 = nn.Linear(input_embedding, num_classes)

    def forward(self, x):
        x = self.attention(x)
        x = self.residual_block3(x)
        x = self.residual_block4(x)
        embedding = x
        output = self.fc2(embedding)
        return embedding, output

# 注意力层和残差块的定义与原模型相同
class SelfAttention(nn.Module):
    def __init__(self, feature_dim):
        super(SelfAttention, self).__init__()
        self.feature_dim = feature_dim
        self.query_weight = nn.Parameter(torch.randn(feature_dim, feature_dim))
        self.key_weight = nn.Parameter(torch.randn(feature_dim, feature_dim))
        self.value_weight = nn.Parameter(torch.randn(feature_dim, feature_dim))
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x):
        x = self.norm(x)
        Q = self.norm(torch.matmul(x, self.query_weight))
        K = self.norm(torch.matmul(x, self.key_weight))
        V = self.norm(torch.matmul(x, self.value_weight))

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.feature_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        weighted_values = torch.matmul(attention_weights, V)
        return weighted_values

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.selu = nn.SELU()
        self.fc_residual = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        residual = self.fc_residual(x)
        x = self.fc(x)
        x = self.selu(x)
        x += residual
        return x
