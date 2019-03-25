import argparse
import time
import collections
import os
import sys
import torch
import torch.nn
from torch.autograd import Variable
import torch.nn as nn
import numpy
import re
import matplotlib.pyplot as plt
from prettytable import PrettyTable

np = numpy


parser = argparse.ArgumentParser(description='PyTorch Penn Treebank Language Modeling')



#arguments for generating samples
parser.add_argument('--model_dir', type=str, default='',
                     help='Model directory including file called best_params.pt')

args = parser.parse_args()
argsdict = args.__dict__
argsdict['code_file'] = sys.argv[0]

model_path_dir = args.model_dir
if model_path_dir!='':
    model_path = os.path.join(model_path_dir, 'best_params.pt')
    if not(os.path.exists(model_path)):
        raise Exception('Model file doesn\'t exist at {}'.format(model_path))
    
else:
    raise Exception('You must enter the saved model dir --model_dir')

# Loading learning curves

curve_data = np.load(os.path.join(model_path_dir, 'learning_curves.npy'))

train_losses = curve_data.item()['train_losses'] 
val_losses = curve_data.item()['val_losses'] 

train_ppls = curve_data.item()['train_ppls'] 
val_ppls = curve_data.item()['val_ppls'] 

# now loading model params and updating args with the model params
model_params = {}
with open (os.path.join(model_path_dir, 'exp_config.txt'), 'r') as params_file:
    for line in params_file:
        line = line.strip().split('    ')
        try:
            if '.' in line[1]:
                model_params[line[0]] = float(line[1])
            else:
                model_params[line[0]] = int(line[1])

        except:
            model_params[line[0]] = line[1]


n_steps_per_epoch = len(train_losses) // model_params['num_epochs']

# checking wall clock time per epoch and interpolating the missing data from the log file
wall_clock_time = []
epochs = []
with open (os.path.join(model_path_dir, 'log.txt'), 'r') as log_file:
    epoch_num = 0
    for line in log_file:
        line = line.strip()
        time_per_epoch = float(re.findall(r'[0-9]+\.?[0-9]*$',line)[0])
        start_time = 0
        if len(wall_clock_time)>0:
            start_time += wall_clock_time[-1]
        wall_clock_time.append(time_per_epoch+start_time)
        epochs.append(epoch_num)
        epoch_num +=1


plt.style.use('ggplot')

# plot train/val loss vs epoch 
plt.plot(epochs, train_ppls )
plt.plot(epochs, val_ppls )
plt.legend(['Train PPl', 'Val PPl'])
plt.xlabel('Epochs')
plt.ylabel('Perplexity')
plt.savefig(os.path.join(model_path_dir,'perplexity_epochs.png'))
plt.close()
plt.clf()

plt.plot(wall_clock_time, train_ppls )
plt.plot(wall_clock_time, val_ppls )
plt.legend(['Train PPl', 'Val PPl'])
plt.xlabel('Wall Clock time (s)')
plt.ylabel('Perplexity')
plt.savefig(os.path.join(model_path_dir,'perplexity_wall_clock.png'))


params = ['model','num_layers','optimizer','initial_lr','seq_len','dp_keep_prob','batch_size','hidden_size','emb_size','num_epochs']
model_summary_table = PrettyTable(params+['Train PPl','Val PPl'])
model_summary_table.add_row([model_params[param] for param in params]+[train_ppls[-1],val_ppls[-1]])

with open (os.path.join(model_path_dir, 'result_table.txt'), 'w') as result:
    result.write(model_summary_table.get_string())
    
