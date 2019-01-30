import time
import argparse
import numpy as np
from train import train
import pickle as pkl
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from args import get_text_args
from utils import *
from train import train_linear, adj, sp_adj, label_dict, index_dict
import torch.nn.functional as F
from models import get_model
from math import log

args = get_text_args()
set_seed(args.seed, args.cuda)

adj_dense = sparse_to_torch_dense(sp_adj, device='cpu')
feat_dict, precompute_time = sgc_precompute(adj, adj_dense, args.degree-1, index_dict)
if args.dataset == "mr": nclass = 1
else: nclass = label_dict["train"].max().item()+1

def linear_objective(space):
    model = get_model(args.model, nfeat=feat_dict["train"].size(1),
                      nclass=nclass,
                      nhid=0, dropout=0, cuda=args.cuda)
    val_acc, _, _ = train_linear(model, feat_dict, space['weight_decay'], args.dataset=="mr")
    print( 'weight decay ' + str(space['weight_decay']) + '\n' + \
          'overall accuracy: ' + str(val_acc))
    return {'loss': -val_acc, 'status': STATUS_OK}

# Hyperparameter optimization
space = {'weight_decay' : hp.loguniform('weight_decay', log(1e-6), log(1e-0))}

best = fmin(linear_objective, space=space, algo=tpe.suggest, max_evals=60)
print(best)

with open('{}.SGC.tuning.txt'.format(args.dataset), 'w') as f:
    f.write(str(best['weight_decay']))
