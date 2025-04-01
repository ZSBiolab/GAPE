import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from age_pre_model import AgeModel  
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import matplotlib.pyplot as plt

# check gpu available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# load data
df_train = pd.read_csv('data/train_data_demo.csv')
df_pre = pd.read_csv('data/train_data_demo.csv')

results_path = 'results'

selected_data_train = df_train.iloc[1:, 1:]  
selected_data_pre = df_pre.iloc[1:, 1:]  

# load label and data
labels_train = df_train.iloc[0, 1:]  
labels_pre = df_pre.iloc[0, 1:]  
data_train = selected_data_train.T  # Transpose so that each line becomes a sample.
data_pre = selected_data_pre.T  # Transpose so that each line becomes a sample.


label_encoder = LabelEncoder()
encoded_labels_train = label_encoder.fit_transform(labels_train)
print(f"Total have {len(set(encoded_labels_train))} label。")


X = torch.tensor(data_pre.values.astype(float)).float().to(device)


dataset = TensorDataset(X)
data_loader = DataLoader(dataset, batch_size=8)
input_embedding = 512


model_path = os.path.join(results_path, 'run/Age_model.pt')
model = AgeModel(X.shape[1], input_embedding, len(set(encoded_labels_train))).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# predict
all_outputs = []
all_embedding = []
with torch.no_grad():
    for inputs, in data_loader:
        inputs = inputs.to(device)
        embedding, outputs = model(inputs)
        all_outputs.extend(outputs.cpu().numpy())
        all_embedding.extend(embedding.cpu().numpy())

# check data match
all_embedding = np.array(all_embedding).T  
if all_embedding.shape[1] != len(labels_pre):
    raise ValueError(f"Shape mismatch: all_embedding has {all_embedding.shape[1]} samples, but labels_pre has {len(labels_pre)} labels")

# Print format is DataFrame, and make sure the column names are unique.
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

unique_labels_pre = make_unique(labels_pre.astype(str))  
output_df = pd.DataFrame(all_embedding, columns=unique_labels_pre)

# Remove suffixes from column names
def remove_suffix(label):
    parts = label.split('.')
    return parts[0]

clean_labels = [remove_suffix(str(label)) for label in output_df.columns]  

output_df.columns = clean_labels
output_df.to_csv(os.path.join(results_path, 'run/output_clinic_data_embedding.csv'), index=False)
print("Outputs are saved to 'results_aus_A_maxmin_Our/run/output_train_data_embedding.csv'.")

# Convert output to label
predicted_labels = label_encoder.inverse_transform(np.argmax(all_outputs, axis=1))
# Decode the real tag from the original tag.
original_labels_pre = label_encoder.inverse_transform(label_encoder.transform(labels_pre))

df_predictions = pd.DataFrame({
    'True Label': original_labels_pre,
    'Predicted Label': predicted_labels
})


df_predictions['Mismatch'] = (df_predictions['True Label'] != df_predictions['Predicted Label']).astype(bool)
df_predictions['Mismatch'] = ~df_predictions['Mismatch']  # Invert to make the error False and the correct one True.


accuracy = df_predictions['Mismatch'].mean()
print(f"pre Accuracy: {accuracy:.2%}")
df_predictions.to_csv(os.path.join(results_path, 'run/pre_age.csv'), index=False)
print("Predictions are saved to 'results_aus_A_maxmin_Our/run/pre_age.csv'.")

# Reload the generated pre-age file and visualize it.
df_result = pd.read_csv(os.path.join(results_path, 'run/pre_age.csv'))

plt.figure(figsize=(12, 8))
plt.grid(True, linestyle='--', alpha=0.7)

# Sets the transparency of the marker point.
plt.scatter(df_result.index, df_result['True Label'], color='blue', marker='^', label='Real Age', alpha=0.6, edgecolors='w', s=100)
plt.scatter(df_result.index, df_result['Predicted Label'], color='red', marker='o', label='Predicted Age', alpha=0.6, edgecolors='w', s=100)

# Mark the same value
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
