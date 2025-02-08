import torch

def attributeResponse(inputs, index, categories):
    categories = torch.tensor(categories)

    repeated_inputs = inputs.unsqueeze(0).expand(len(categories), *inputs.shape).clone()
    repeated_inputs[torch.arange(len(categories)), :, index] = categories.unsqueeze(-1)

    return repeated_inputs.view(-1, *inputs.shape[1:])