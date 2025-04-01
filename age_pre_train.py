import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,  f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from age_pre_model import AgeModel 
from tqdm import tqdm  
import numpy as np
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


df = pd.read_csv('data/train_data_demo.csv')

selected_data = df.iloc[1:, 1:]


labels = df.iloc[0, 1:]  
data = selected_data.T 


label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)


print(f"total have {len(set(encoded_labels))} labels.")


X = torch.tensor(data.values.astype(float)).float().to(device)
y = torch.tensor(encoded_labels).long().to(device)


dataset = TensorDataset(X, y)
data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

num_epochs = 200  
input_embedding = 512
save_epochs = num_epochs-20


model = AgeModel(X.shape[1], input_embedding, len(set(encoded_labels))).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

epoch_accuracies = []
best_accuracy = 0
best_epoch_labels = []
best_epoch_predictions = []
epoch_f1_scores = []  
best_epoch = -1
best_outputs = []  
best_labels = []  
best_embedding = []
best_model_path = 'run/Age_model.pt'

# Check if results folder exists
result_path = 'results'
if not os.path.exists(result_path):
    os.makedirs(result_path)

for epoch in tqdm(range(num_epochs), desc='Training Epochs'):
    total = 0
    correct = 0
    current_epoch_loss = 0
    current_epoch_labels = []
    current_epoch_predictions = []
    model.train()
    epoch_outputs = []  
    epoch_labels = []  
    epoch_embedding = []
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        embedding, outputs = model(inputs)
        loss = criterion(outputs, labels)
        current_epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        
        softmax_outputs = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(softmax_outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        current_epoch_labels.extend(labels.cpu().numpy())
        current_epoch_predictions.extend(predicted.cpu().numpy())
        epoch_outputs.append(outputs.detach().cpu().numpy())
        epoch_embedding.append(embedding.detach().cpu().numpy())
        epoch_labels.append(labels.cpu().numpy())

    accuracy = 100 * correct / total
    epoch_accuracies.append(accuracy)
   
    current_f1_score = f1_score(current_epoch_labels, current_epoch_predictions, average='macro')
    epoch_f1_scores.append(current_f1_score)
    avg_epoch_loss = current_epoch_loss / len(data_loader)

    if (epoch + 1) % 1 == 0:
        tqdm.write(
            f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}, Accuracy: {accuracy:.2f}%, F1 Score: {current_f1_score:.4f}')

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_epoch_labels = current_epoch_labels
        best_epoch_predictions = current_epoch_predictions
        best_epoch = epoch
        best_outputs = np.vstack(epoch_outputs)
        best_embedding = np.vstack(epoch_embedding)
        best_labels = np.hstack(epoch_labels)  
       
        if epoch > save_epochs:
            if not os.path.exists('run'):
                os.makedirs('run')
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model with accuracy {best_accuracy:.2f}% at epoch {best_epoch} to {best_model_path}")

#scheduler.step()


original_best_labels = label_encoder.inverse_transform(best_labels)


original_best_epoch_labels = label_encoder.inverse_transform(best_epoch_labels)
original_best_epoch_predictions = label_encoder.inverse_transform(best_epoch_predictions)

if not os.path.exists('run'):
    os.makedirs('run')

df_predictions = pd.DataFrame({
    'True Label': original_best_epoch_labels,
    'Predicted Label': original_best_epoch_predictions
}).transpose()

df_predictions.to_csv('run/best_epoch_predictions.csv', header=False, index=False)
print("Predictions of the best accuracy epoch are saved to 'best_epoch_predictions.csv'")


if best_epoch >= 0:
    combined = np.column_stack((original_best_labels, best_embedding))
    df_best = pd.DataFrame(combined)
    df_best.to_csv(f'run/best_epoch_{best_epoch}_outputs.csv', header=False, index=False)
    print(f"Best epoch ({best_epoch}) outputs and labels are saved to 'run/best_epoch_{best_epoch}_outputs.csv'.")
else:
    print("No best epoch found.")

# Save accuracy and F1 score data to CSV
pd.DataFrame({'Average Accuracy': epoch_accuracies}).to_csv(f'{result_path}/average_accuracies.csv', index=False)
pd.DataFrame({'Average F1 Score': epoch_f1_scores}).to_csv(f'{result_path}/average_f1_scores.csv', index=False)


plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1, 1), epoch_accuracies, label='Average Accuracy per Epochs')
plt.xlabel('Epoch')
plt.ylabel('Average Accuracy')
plt.title('Average Accuracy Over Epochs')
plt.legend()
plt.savefig(f'{result_path}/average_accuracy.pdf', dpi=600, format='pdf')
plt.close()


best_confusion_matrix = confusion_matrix(best_epoch_labels, best_epoch_predictions)
original_labels = label_encoder.inverse_transform(range(len(label_encoder.classes_)))
plt.figure(figsize=(10, 8))
sns.heatmap(best_confusion_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=original_labels, yticklabels=original_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
#plt.title(f'Confusion Matrix at Best Accuracy (Epoch {epoch_accuracies.index(best_accuracy) + 1})')
plt.title(f'Confusion Matrix')
plt.savefig(f'{result_path}/confusion_matrix.pdf', dpi=600, format='pdf')
plt.close()
pd.DataFrame(best_confusion_matrix, index=label_encoder.classes_, columns=label_encoder.classes_).to_csv(f'{result_path}/confusion_matrix.csv')


plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1, 1), epoch_f1_scores, label='Average F1 Score per Epochs', color='red')
plt.xlabel('Epoch')
plt.ylabel('Average F1 Score')
plt.title('Average F1 Score Over Epochs')
plt.legend()
plt.savefig(f'{result_path}/f1_score.pdf', dpi=600, format='pdf')
plt.close()
