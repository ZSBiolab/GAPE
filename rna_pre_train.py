import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from rna_pre_model import prernamodel  
from tqdm import tqdm 
import os
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


df = pd.read_csv('data/train_data_demo.csv')
df_embedding = pd.read_csv('run/output_clinic_data_embedding.csv')

df_label_selected_data = df.iloc[1:, 1:]
df_input_selected_data = df.iloc[1:, 1:]

pre_age_labels = df.iloc[0, 1:]  
pre_age_label_encoder = LabelEncoder()
encoded_pre_age_labels = pre_age_label_encoder.fit_transform(pre_age_labels)
print(f"total have {len(set(encoded_pre_age_labels))} labels.")
pre_age_labels = torch.tensor(encoded_pre_age_labels).long().to(device)
print(f'pre_age_labels shape: {pre_age_labels.shape}')



inputs = df_label_selected_data.values.astype(float)
embedding = df_embedding.values.astype(float)
labels = df_label_selected_data.values.astype(float)


inputs = torch.tensor(inputs).float().t().to(device)
labels = torch.tensor(labels).float().t().to(device)
embedding = torch.tensor(embedding).float().t().to(device)

print("inputs shape:", inputs.shape)
print("labels shape:", labels.shape)
print("embedding shape:", embedding.shape)
embedding_size = 512


dataset = TensorDataset(inputs, labels, embedding, pre_age_labels)
data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

pre_rna_model = prernamodel(inputs.shape[1], labels.shape[1], embedding_size, len(set(encoded_pre_age_labels))).to(device)
num_epochs = 1000 


pretrained_weights_path = 'run/Age_model.pt'
pretrained_dict = torch.load(pretrained_weights_path, map_location=device)
model_dict = pre_rna_model.state_dict()


for name, param in pretrained_dict.items():
    if 'residual_block3' in name or 'residual_block4' in name or 'fc2' in name or 'attention2' in name:
        if name in model_dict:
            model_dict[name].data.copy_(param.data)


pre_rna_model.load_state_dict(model_dict)

for name, param in pre_rna_model.named_parameters():
    if 'residual_block3' in name or 'residual_block4' in name or 'fc2' in name or 'attention2' in name:
        param.requires_grad = False


optimizer = optim.AdamW(filter(lambda p: p.requires_grad, pre_rna_model.parameters()), lr=0.00001)
criterion_pre_age = nn.CrossEntropyLoss()
def criterion_pre_rna(output, target, alpha=0.8):
    mse_loss = nn.MSELoss()
    cosine_similarity = nn.CosineSimilarity(dim=1)

    mse = mse_loss(output, target)
    cosine_sim = cosine_similarity(output, target)
    cosine_loss = 1 - cosine_sim.mean()  

    combined = alpha * mse + (1 - alpha) * cosine_loss
    return combined


result_path = 'results'
if not os.path.exists(result_path):
    os.makedirs(result_path)


log_file = os.path.join(result_path, 'training_log.txt')
with open(log_file, 'w') as log:
    log.write("Epoch,Loss,Pre RNA Loss 1,Pre RNA Loss 2,Time (s)\n")


best_loss = float('inf')
best_model_path = 'run/RNA_model_best.pt'


for epoch in tqdm(range(num_epochs), desc='Training Epochs'):
    start_time = time.time()
    total_loss = 0
    total_pre_rna_loss1 = 0
    total_pre_rna_loss2 = 0
    pre_rna_model.train()
   
    torch.cuda.empty_cache() 

    for inputs, labels, embedding, pre_age_labels in data_loader:
        inputs, labels, embedding, pre_age_labels = inputs.to(device), labels.to(device), embedding.to(device), pre_age_labels.to(device)
        mask = (torch.rand(inputs.shape) > 0.05).float().to(device)
        pre_rna_inputs_masked = inputs * mask
        optimizer.zero_grad()
        pre_rna_outputs1, pre_rna_outputs2 = pre_rna_model(pre_rna_inputs_masked, embedding)
        pre_rna_loss1 = criterion_pre_rna(pre_rna_outputs1, labels)
        pre_rna_loss2 = criterion_pre_age(pre_rna_outputs2, pre_age_labels)
        pre_rna_loss = pre_rna_loss1 + pre_rna_loss2
        pre_rna_loss.backward()
        optimizer.step()
        total_loss += pre_rna_loss.item()
        total_pre_rna_loss1 += pre_rna_loss1.item()
        total_pre_rna_loss2 += pre_rna_loss2.item()

    avg_epoch_loss = total_loss / len(data_loader)
    avg_pre_rna_loss1 = total_pre_rna_loss1 / len(data_loader)
    avg_pre_rna_loss2 = total_pre_rna_loss2 / len(data_loader)

    end_time = time.time()
    epoch_time = end_time - start_time
    
    with open(log_file, 'a') as log:
        log.write(
            f"{epoch + 1},{avg_epoch_loss:.4f},{avg_pre_rna_loss1:.4f},{avg_pre_rna_loss2:.4f},{epoch_time:.2f}\n")

    
    mem_allocated = torch.cuda.memory_allocated()
    mem_reserved = torch.cuda.memory_reserved()
    tqdm.write(f'Epoch [{epoch + 1}/{num_epochs}], \nLoss: {avg_epoch_loss:.4f},'
               f' Pre RNA Loss: {avg_pre_rna_loss1:.4f}, Pre RNA Loss 2: {avg_pre_rna_loss2:.4f}, '
               f'Memory Allocated: {mem_allocated} bytes, Memory Reserved: {mem_reserved} bytes')
    
    if epoch >= 950:
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(pre_rna_model.state_dict(), best_model_path)
            print(f"Saved best model at epoch {epoch + 1} with loss {best_loss:.4f}")

        
    if epoch == num_epochs - 1:
        torch.save(pre_rna_model.state_dict(), 'run/RNA_model.pt')
        print(f"Saved final model at epoch {epoch + 1}")

