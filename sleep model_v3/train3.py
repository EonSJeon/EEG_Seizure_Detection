import os
import numpy as np
from model3 import SimpleMLP
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
import numpy as np
import torch.nn as nn

import seaborn as sns  # Import seaborn for heatmap creation


root_data_path = '/Users/jeonsang-eon/sleep_data_processed3/'

sub_nums = [
    2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 17, 18, 19, 20,
    21, 22, 23, 25, 26, 27, 28, 29, 30,
    31, 32, 33
]

class SubDataset:
    def __init__(self, data, labels):
        # Reshape data to be (samples, features)
        # Combining the dimensions: channels * features * time_steps
        self.data = data.reshape(data.shape[0], -1)
        self.labels = labels

# Redefining the datasets list to include reshaped data
datasets = []

for sub_num in sub_nums:
    sub_name = f"sub-{sub_num:02d}"
    data_path = os.path.join(root_data_path, sub_name, f"{sub_name}_features.npy")
    label_path = os.path.join(root_data_path, sub_name, f"{sub_name}_labels.npy")
    
    # Load and reshape data and labels
    data = np.load(data_path)
    labels = np.load(label_path)
    
    # Create an instance of SubDataset with reshaped data
    sub_dataset = SubDataset(data, labels)
    datasets.append(sub_dataset)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Assuming you have already defined 'SubDataset' and loaded your datasets

num_epochs = 100
batch_size = 64

def create_dataloader(datasets, index, batch_size):
    train_data = np.concatenate([ds.data for j, ds in enumerate(datasets) if j != index], axis=0)
    train_labels = np.concatenate([ds.labels for j, ds in enumerate(datasets) if j != index], axis=0)
    train_labels_indices = np.argmax(train_labels, axis=1)
    train_tensor = torch.tensor(train_data, dtype=torch.float32)
    label_tensor = torch.tensor(train_labels_indices, dtype=torch.long)
    return DataLoader(TensorDataset(train_tensor, label_tensor), batch_size=batch_size, shuffle=True)


def plot_confusion_matrix(cm, sub_num, labels=['Class 0', 'Class 1']):
    cm = cm.astype(int)  # Ensure the matrix contains integers for correct annotation
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix for {sub_num}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'Confusion_Matrix_{sub_num}.png')
    plt.close()

def plot_roc_curve(fpr, tpr, roc_auc, i):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic for Subject {i}')
    plt.legend(loc="lower right")
    plt.savefig(f'ROC_Curve_Subject_{i}.png')
    plt.close()

def plot_loss_curve(epoch_losses, sub_num):
    plt.figure()
    plt.plot(range(len(epoch_losses)), epoch_losses, marker='o', linestyle='-', color='blue')
    plt.title(f'Loss Curve for Subject {sub_num}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(f'Loss_Curve_Subject_{sub_num}.png')
    plt.close()

total_conf_matrix = np.zeros((2, 2))  # Adjust dimensions based on the number of classes
roc_aucs = []

for i, test_dataset in enumerate(datasets):
    model = SimpleMLP()  # Ensure your model's input layer matches the number of features
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loader = create_dataloader(datasets, i, batch_size)

    epoch_losses = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(train_loader)
        epoch_losses.append(average_loss)

    # Save the loss data and plot
    sub_num = sub_nums[i]  # Correctly associate sub_num with the loop variable
    np.savetxt(f'Loss_Subject_{sub_num}.txt', np.array(epoch_losses), fmt='%.4f')
    plot_loss_curve(epoch_losses, sub_num)

    model.eval()
    test_inputs = torch.tensor(test_dataset.data, dtype=torch.float32)
    test_labels = np.argmax(test_dataset.labels, axis=1)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)
    with torch.no_grad():
        raw_predictions = model(test_inputs)
        predictions = nn.functional.softmax(raw_predictions, dim=1)
        pred_prob = predictions.numpy()
        pred_classes = pred_prob.argmax(axis=1)

        conf_mat = confusion_matrix(test_labels, pred_classes)
        total_conf_matrix += conf_mat/len(test_labels)
        plot_confusion_matrix(conf_mat, sub_num)
        
        fpr, tpr, _ = roc_curve(test_labels, pred_prob[:, 1], pos_label=1)
        roc_auc = auc(fpr, tpr)
        roc_aucs.append(roc_auc)
        plot_roc_curve(fpr, tpr, roc_auc, sub_num)

# Compute the average confusion matrix and scale it by 100
average_conf_matrix = total_conf_matrix / len(datasets) * 100
print(average_conf_matrix)
average_conf_matrix = average_conf_matrix.astype(int)  # Convert to integer for visualization and saving

plot_confusion_matrix(average_conf_matrix, 'Average')  # Save the average confusion matrix as a heatmap
np.savetxt('Average_Confusion_Matrix.txt', average_conf_matrix, fmt='%d')  # Also save as text for numerical analysis

average_roc_auc = sum(roc_aucs) / len(roc_aucs)
print(f"Average AUC: {average_roc_auc}")