import torch
import math

def stochastic_response(column, epsilon, categories):
    categories = torch.tensor(categories)
    K = len(categories)
    p = math.exp(epsilon) / (math.exp(epsilon) + K - 1)

    prob_matrix = torch.full_like(column.unsqueeze(-1).expand(*column.shape, K), (1 - p) / (K - 1))

    correct_indices = (column.unsqueeze(-1) == categories.view((1,) * column.dim() + (-1,)))
    prob_matrix[correct_indices] = p

    perturbed_indices = torch.multinomial(prob_matrix, 1).squeeze()
    perturbed_column = categories[perturbed_indices]
    return perturbed_column

def attributeStochastic(inputs, index, epsilon, categories):
    column = inputs[:, index]
    inputs = inputs.clone()
    inputs[:, index] = stochastic_response(column, epsilon, categories)
    return inputs
    