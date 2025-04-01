import torch
import torch.nn as nn
import torch.nn.functional as F

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
        x = x + residual
        return x

class prernamodel(nn.Module):
    def __init__(self, input_features, output_feature, input_embedding, num_classes):
        super(prernamodel, self).__init__()

        combine_features = input_features+input_embedding

        self.residual_block1 = ResidualBlock(combine_features, 1024)
        self.residual_block2 = ResidualBlock(1024, output_feature)
        self.attention1 = SelfAttention(output_feature)
        self.attention2 = SelfAttention(output_feature)

        self.ac = nn.Sigmoid()

        self.residual_block3 = ResidualBlock(output_feature, 1024)
        self.residual_block4 = ResidualBlock(1024, input_embedding)
        self.fc2 = nn.Linear(input_embedding, num_classes)


    def forward(self, input_rna, input_embedding):
        # Concatenate x1 and x2 along the feature dimension
        #print(f'input_rna shape: {input_rna.shape}')
        #print(f'input_embedding shape: {input_embedding.shape}')
        x = torch.cat((input_rna, input_embedding), dim=-1)
        # Attention and further processing
        x = self.residual_block1(x)
        #x = self.residual_block2(x)
        x = self.residual_block2(x)
        x = self.attention1(x)
        rna_output = self.ac(x)

        x1 = self.attention2(rna_output)
        x1 = self.residual_block3(x1)
        embedding = self.residual_block4(x1)
        age_output = self.fc2(embedding)
        return rna_output, age_output

    def load_pretrained_weights(self, path):
        pretrained_dict = torch.load(path)
        model_dict = self.state_dict()
        # 过滤出需要的参数
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        # 冻结特定部分的参数
        for name, param in self.named_parameters():
            if 'residual_block3' in name or 'residual_block4' in name or 'fc2' in name:
                param.requires_grad = False