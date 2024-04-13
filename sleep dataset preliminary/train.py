import sympy as sp


# Dictionary of label distributions
LABEL_DSTR = {
    'W': 3114, '1': 2105, '2': 1243, '3': 42,
    '1 (unscorable)': 6, '2 (unscorable)': 10, '3 (unscorable)': 11, '2 or 3 (unscorable)': 3, 
    'W (uncertain)': 38, '1 (uncertain)': 41, '2 (uncertain)': 8, '3 (uncertain)': 1,
    'unscorable': 2, 'Unscorable': 1
}

# Labels considered for basic prior calculation
CERTAIN_LABELS = ['W', '1', '2', '3']

# Calculate the sum of counts for certain labels
tempSum = sum(LABEL_DSTR[label] for label in CERTAIN_LABELS)
PRIOR = {label: LABEL_DSTR[label] / tempSum for label in CERTAIN_LABELS}

def uncertain_vec(label):
    """ Calculates transformed probabilities for given label considering the effect of other labels. """
    p = sp.symbols('p')
    p_label = PRIOR[label]
    other_labels = {k: v for k, v in PRIOR.items() if k != label}
    max_other_p = max(other_labels.values())

    # Solve the probability transformation equation
    equation = sp.Eq(p, (1 - p) / (1 - p_label) * max_other_p)
    solution = sp.solve(equation, p)
    p_value = min([sol.evalf() for sol in solution if sol.is_real and sol >= 0], default=0)

    # Apply the transformation to all probabilities
    transformed_probabilities = {k: (v * (1 - p_value) / (1 - p_label) if k != label else p_value) for k, v in PRIOR.items()}
    return [transformed_probabilities[k] for k in CERTAIN_LABELS]

# Dictionary for storing the solution vectors for each label type
SOL_DICT = {
    'W': [1, 0, 0, 0],
    '1': [0, 1, 0, 0],
    '2': [0, 0, 1, 0],
    '3': [0, 0, 0, 1],
    'unscorable': list(PRIOR.values()), 
    'Unscorable': list(PRIOR.values())
}

# Update dictionary with vectors from uncertain vector calculations
for label in CERTAIN_LABELS:
    uncertain_vec_result = uncertain_vec(label)
    for uncertain_label in [f'{label} (unscorable)', f'{label} (uncertain)']:
        SOL_DICT[uncertain_label] = uncertain_vec_result

# Handle the special case '2 or 3 (unscorable)'
SOL_DICT['2 or 3 (unscorable)'] = [0, 0, 0.5, 0.5]

#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################

# SOL_DICT Initiated

import torch.optim as optim

criterion = nn.CrossEntropyLoss()  # Appropriate for classification with a fixed number of classes
optimizer = optim.Adam(model.parameters(), lr=0.001)

from torch.utils.data import Dataset, DataLoader

class EEGDataset(Dataset):
    def __init__(self, data, labels):
        """
        Args:
            data (list or ndarray): The data samples.
            labels (list): The target labels.
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label

# Example: Loading dataset
# data should be preprocessed to have shape [batch_size, seq_length, channels, height, width]
# labels should be indices for classification: [batch_size]
train_data = [...]  # Populate with your data
train_labels = [...]  # Populate with your labels
train_dataset = EEGDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

def train_model(model, train_loader, criterion, optimizer, num_epochs=25):
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)  # Assume 'device' is defined (either 'cuda' or 'cpu')
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    print('Finished Training')

# Train the model
train_model(model, train_loader, criterion, optimizer, num_epochs=50)
