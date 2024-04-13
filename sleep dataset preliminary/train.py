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

print("Original Priors:", PRIOR)
print("Transformed Probabilities:", SOL_DICT)

import prettytable as pt

# Create a table with headings
table = pt.PrettyTable()
table.field_names = ["Label", "W", "1", "2", "3"]

# Populate the table with data from SOL_DICT
for label, probabilities in SOL_DICT.items():
    # Convert each probability list into a string for nice formatting (optional)
    probabilities_str = [f"{prob:.5f}" for prob in probabilities]
    table.add_row([label] + probabilities_str)

print(table)
