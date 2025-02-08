import torch
import torch.nn as nn
from torch.nn import functional as F

from response import attributeResponse
from stochastic import attributeStochastic

def binary_classification_step(net, batch, batch_idx, **kw):
	inputs, labels  = batch
	inputs, labels = inputs.to(kw['device']), labels.to(kw['device'])

	scores = net(inputs)
	loss = F.binary_cross_entropy_with_logits(scores, labels, reduction = 'sum')

	correct = ((scores > 0) == labels).sum()
	return {'loss':loss, 'correct':correct}

def predict_binary_classification_step(net, batch, batch_idx, **kw):
	inputs, _ = batch
	inputs = inputs.to(kw['device'])
	scores = net(inputs)

	return {'predictions':(scores > 0)}

def response_step(net, batch, batch_idx, **kw):

	inputs, labels  = batch
	inputs = attributeResponse(inputs, index=8, categories = [-1, 1.])
	labels = labels.repeat(2, 1)
	inputs, labels = inputs.to(kw['device']), labels.to(kw['device'])

	scores = net(inputs)
	loss = F.binary_cross_entropy_with_logits(scores, labels, reduction = 'sum') / 2

	correct = ((scores > 0) == labels).sum()

	loss_ = loss / kw['batch_size']
	loss_.backward()

	return {'loss':loss,
			'correct':correct
	}


def stochastic_step(net, batch, batch_idx, **kw):

	inputs, labels  = batch
	inputs = attributeRR(inputs, index=8, epsilon=0., categories=[-1, 1.])
	inputs, labels = inputs.to(kw['device']), labels.to(kw['device'])

	scores = net(inputs)
	loss = F.binary_cross_entropy_with_logits(scores, labels, reduction = 'sum')

	correct = ((scores > 0) == labels).sum()

	loss_ = loss / kw['batch_size']
	loss_.backward()

	return {'loss':loss,
			'correct':correct
	}

def erm_step(net, batch, batch_idx, **kw):

	inputs, labels  = batch
	inputs, labels = inputs.to(kw['device']), labels.to(kw['device'])

	scores = net(inputs)
	loss = F.binary_cross_entropy_with_logits(scores, labels, reduction = 'sum')

	correct = ((scores > 0) == labels).sum()

	loss_ = loss / kw['batch_size']
	loss_.backward()

	return {'loss':loss,
			'correct':correct
	}

def binary_fair_classification_step(net, batch, batch_idx, **kw):
	inputs, labels  = batch
	inputs, labels = inputs.to(kw['device']), labels.to(kw['device'])

	scores = net(inputs)
	original_predictions = scores > 0

	flipped_inputs = inputs.clone()
	flipped_inputs[:, 8] = - flipped_inputs[:, 8]

	flipped_scores = net(flipped_inputs)
	flipped_predictions = flipped_scores > 0

	consistent = (original_predictions == flipped_predictions).sum()

	loss = F.binary_cross_entropy_with_logits(scores, labels, reduction = 'sum')
	flipped_loss = F.binary_cross_entropy_with_logits(flipped_scores, labels, reduction = 'sum')

	correct = (original_predictions == labels).sum()
	flipped_correct = (flipped_predictions == labels).sum()
	return {'loss':loss,
			'correct':correct,
			'correct/flip':flipped_correct,
			'loss/flip':flipped_loss,
			'consistent':consistent
	}