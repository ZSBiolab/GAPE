import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('data/train_age_YG_normal.csv')

pre_age_labels = df.iloc[0, 1:]
pre_age_label_encoder = LabelEncoder()
encoded_pre_age_labels = pre_age_label_encoder.fit_transform(pre_age_labels)
print(f"total {len(set(encoded_pre_age_labels))} label")


selected_train_samples = pd.DataFrame()
selected_test_samples = pd.DataFrame()


total_samples = df.shape[1] - 1  
train_samples_count = int(0.8 * total_samples)
test_samples_count = total_samples - train_samples_count


label_groups = {label: [] for label in set(encoded_pre_age_labels)}
for col_index, label in enumerate(encoded_pre_age_labels):
    label_groups[label].append(col_index + 1)  


samples_drawn = 0
while samples_drawn < train_samples_count:
    for label in sorted(label_groups.keys()):
        if samples_drawn >= train_samples_count:
            break
        if label_groups[label]:
            selected_col = label_groups[label].pop(0)
            selected_train_samples = pd.concat([selected_train_samples, df.iloc[:, selected_col]], axis=1)
            samples_drawn += 1


for label, cols in label_groups.items():
    for col in cols:
        selected_test_samples = pd.concat([selected_test_samples, df.iloc[:, col]], axis=1)


first_column = df.iloc[:, 0]


selected_train_samples.insert(0, df.columns[0], first_column)


selected_test_samples.insert(0, df.columns[0], first_column)


selected_train_samples.to_csv('train_aus.csv', index=False)
selected_test_samples.to_csv('test_aus.csv', index=False)

print(f"save {train_samples_count} sample to 'data/train_aus.csv'")
print(f"save {test_samples_count} sample to 'data/test_aus.csv'")
