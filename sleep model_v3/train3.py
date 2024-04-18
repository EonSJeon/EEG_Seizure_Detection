import os
import numpy as np
from model3 import SimpleMLP
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
import numpy as np
import torch.nn as nn


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

# Now data in each SubDataset instance is reshaped to (samples, features)


num_epochs = 30
batch_size = 64

# Function to convert datasets to DataLoader
def create_dataloader(datasets, index, batch_size):
    train_data = np.concatenate([ds.data for j, ds in enumerate(datasets) if j != index], axis=0)
    train_labels = np.concatenate([ds.labels for j, ds in enumerate(datasets) if j != index], axis=0)
    train_labels_indices = np.argmax(train_labels, axis=1)  # Assuming one-hot encoded labels

    train_tensor = torch.tensor(train_data, dtype=torch.float32)
    label_tensor = torch.tensor(train_labels_indices, dtype=torch.long)

    return DataLoader(TensorDataset(train_tensor, label_tensor), batch_size=batch_size, shuffle=True)

# Training and Evaluation
for i, test_dataset in enumerate(datasets):
    model = SimpleMLP()  # Ensure your model's input layer matches the number of features after reshaping
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loader = create_dataloader(datasets, i, batch_size)

    # Train the model
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluation on test data
    model.eval()
    test_inputs = torch.tensor(test_dataset.data, dtype=torch.float32)
    test_labels_indices = np.argmax(test_dataset.labels, axis=1)
    predictions = model(test_inputs).argmax(dim=1)
    conf_matrix = confusion_matrix(test_labels_indices, predictions.numpy())
    
    print(f"Confusion Matrix for subject {i+1}:")
    print(conf_matrix)

