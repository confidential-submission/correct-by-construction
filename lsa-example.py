import torch
from torch.utils.tensorboard import SummaryWriter

from torchiteration import train_plain as train, validate, predict

from steps import erm_step, response_step, stochastic_step, binary_classification_step, predict_binary_classification_step, binary_fair_classification_step



config = {
	'dataset':'census',
	'training_step':'response_step',
	'batch_size':16,
	'optimizer':'SGD',
	'optimizer_config':{
	},
	'scheduler':'StepLR',
	'scheduler_config':{
		'step_size':2000,
		'gamma':1
	},
	'device':'cuda' if torch.cuda.is_available() else 'cpu',
	'validation_step':'binary_fair_classification_step',
	'sensitive_index':6,
}

model = torch.hub.load('cat-claws/nn', 'simplecnn', convs = [], linears = [7, 64, 32, 16, 8, 4], num_classes = 1).to(config['device'])

with torch.no_grad():
    model.layers[1].weight.zero_()
    model.layers[1].weight += 1e-10

writer = SummaryWriter(comment = f"_{config['dataset']}_{model._get_name()}_{config['training_step']}", flush_secs=10)

for k, v in config.items():
	if k.endswith('_step'):
		config[k] = eval(v)
	elif k == 'optimizer':
		config[k] = vars(torch.optim)[v]([p for p in model.parameters() if p.requires_grad], **config[k+'_config'])
		config['scheduler'] = vars(torch.optim.lr_scheduler)[config['scheduler']](config[k], **config['scheduler_config'])		

import pandas as pd
import numpy as np
df = pd.read_parquet("hf://datasets/cestwc/law-school-admissions/data/train-00000-of-00001.parquet")
df_ = df.drop(['enroll', 'asian', 'black', 'hispanic', 'white', 'missingrace', 'urm'], axis=1)
df_.replace(to_replace=-1, value=np.nan, inplace=True)
data = df_.dropna(axis=0)

X = data.drop(data.columns[[-1]], axis=1).values  # Features
y = data.iloc[:, -1].values   # Labels (target)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X[:, 0:6] = scaler.fit_transform(X[:, 0:6])
X[:, 6] = np.where(X[:, 6] == 0, -1, X[:, 6])


dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).view(-1, 1))
train_set, val_set = torch.utils.data.random_split(dataset, [int(len(X)*0.7), len(X)-int(len(X)*0.7)])

train_loader = torch.utils.data.DataLoader(train_set, num_workers = 4, batch_size = config['batch_size'])
val_loader = torch.utils.data.DataLoader(val_set, num_workers = 4, batch_size = config['batch_size'])


for epoch in range(200):
	if epoch > 0:
		train(model, train_loader = train_loader, epoch = epoch, writer = writer, **config)

	validate(model, val_loader = val_loader, epoch = epoch, writer = writer, **config)

	torch.save(model.state_dict(), 'checkpoints/' + writer.log_dir.split('/')[-1] + f"_{epoch:03}.pt")

print(model)

outputs = predict(model, predict_binary_classification_step, val_loader = val_loader, **config)

print(outputs.keys(), outputs['predictions'])

writer.flush()
writer.close()