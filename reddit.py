from time import perf_counter
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_reddit_data, sgc_precompute, set_seed
from metrics import f1
from models import SGC

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--inductive', action='store_true', default=False,
                    help='inductive training.')
parser.add_argument('--test', action='store_true', default=False,
                    help='inductive training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=2,
                    help='Number of epochs to train.')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--normalization', type=str, default='AugNormAdj',
                   choices=['NormLap', 'Lap', 'RWalkLap', 'FirstOrderGCN',
                            'AugNormAdj', 'NormAdj', 'RWalk', 'AugRWalk', 'NoNorm'],
                   help='Normalization method for the adjacency matrix.')
parser.add_argument('--model', type=str, default="SGC",
                    help='model to use.')
parser.add_argument('--degree', type=int, default=2,
                    help='degree of the approximation.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

set_seed(args.seed, args.cuda)

adj, train_adj, features, labels, idx_train, idx_val, idx_test = load_reddit_data(args.normalization, cuda=args.cuda)
print("Finished data loading.")

model = SGC(features.size(1), labels.max().item()+1)
if args.cuda: model.cuda()
processed_features, precompute_time = sgc_precompute(features, adj, args.degree)
if args.inductive:
    train_features, _ = sgc_precompute(features[idx_train], train_adj, args.degree)
else:
    train_features = processed_features[idx_train]

test_features = processed_features[idx_test if args.test else idx_val]

def train_regression(model, train_features, train_labels, epochs):
    optimizer = optim.LBFGS(model.parameters(), lr=1)
    model.train()
    def closure():
        optimizer.zero_grad()
        output = model(train_features)
        loss_train = F.cross_entropy(output, train_labels)
        loss_train.backward()
        return loss_train
    t = perf_counter()
    for epoch in range(epochs):
        loss_train = optimizer.step(closure)
    train_time = perf_counter()-t
    return model, train_time

def test_regression(model, test_features, test_labels):
    model.eval()
    return f1(model(test_features), test_labels)

model, train_time = train_regression(model, train_features, labels[idx_train], args.epochs)
test_f1, _ = test_regression(model, test_features, labels[idx_test if args.test else idx_val])
print("Total Time: {:.4f}s, {} F1: {:.4f}".format(train_time+precompute_time,
                                                  "Test" if args.test else "Val",
                                                  test_f1))
