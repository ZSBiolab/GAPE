import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from age_pre_model import AgeModel
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


df_train = pd.read_csv('data/train_data.csv')
input_embedding = 512


feature_names = df_train.iloc[1:, 0].values


selected_data = df_train.iloc[1:, 1:-1].astype(float)
data = selected_data.T  


labels = df_train.iloc[0, 1:-1].values 
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_classes = len(set(encoded_labels))
print(f"Total have {num_classes} labels.")


X = torch.tensor(data.values, dtype=torch.float32).to(device)
y = torch.tensor(encoded_labels, dtype=torch.long).to(device)


input_dim = X.shape[1]
model = AgeModel(input_dim, input_embedding, num_classes).to(device)
model_path = 'run/Age_model.pt'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


class WrappedModel(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
    def forward(self, x):
        _, output = self.base_model(x)
        return output

wrapped_model = WrappedModel(model)

# Captum Integrated Gradients
ig = IntegratedGradients(wrapped_model)
attr_list = []

for i in range(len(X)):
    sample = X[i].unsqueeze(0)
    label = y[i].item()
    attributions, _ = ig.attribute(sample, target=label, return_convergence_delta=True)
    attr_list.append(attributions.cpu().detach().numpy().flatten())


attributions = np.array(attr_list)
feature_importance = np.mean(np.abs(attributions), axis=0)


num_features = 100
top_features_idx = np.argsort(feature_importance)[-num_features:]


fig, ax = plt.subplots(figsize=(8, 30))
y_ticks = [feature_names[idx] for idx in top_features_idx]
y_positions = range(len(top_features_idx))
importance_scores = feature_importance[top_features_idx]

cmap = plt.get_cmap("viridis_r")
normalize = plt.Normalize(vmin=min(importance_scores), vmax=max(importance_scores))
colors = [cmap(normalize(value)) for value in importance_scores]

ax.scatter(importance_scores, y_positions, color=colors, s=40)
ax.plot(importance_scores, y_positions, color='red', linestyle='-', linewidth=0.9, alpha=0.5)
ax.set_yticks(y_positions)
ax.set_yticklabels(y_ticks, fontsize=10)
ax.set_xlabel("Importance Score")
ax.set_ylabel("Features")
ax.set_title("Top 100 Feature Importances")
ax.invert_yaxis()
ax.grid(True, linestyle='--', alpha=0.6)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=normalize)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.1)
cbar.set_label("Importance Score")

plt.show()
print("The plot of the top 100 important features has been displayed.")


output_df = pd.DataFrame({
    'Feature': [feature_names[idx] for idx in top_features_idx],
    'Importance': importance_scores
}).sort_values(by='Importance', ascending=False)

output_df.to_csv('top_100_attribution_data.csv', index=False)
print("Top 100 attribution data saved as 'top_100_attribution_data.csv'.")
