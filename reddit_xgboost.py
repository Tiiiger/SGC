from time import perf_counter
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_reddit_data, sgc_precompute, set_seed
from metrics import f1
from models import SGC
import xgboost as xgb
from sklearn.metrics import accuracy_score

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

adj, train_adj, features, labels, idx_train, idx_val, idx_test = load_reddit_data(args.normalization)
print("Finished data loading.")

nclass = labels.max().item()+1
model = SGC(features.size(1), nclass)
if args.cuda: model.cuda()
processed_features, precompute_time = sgc_precompute(features, adj, args.degree)
if args.inductive:
    train_features, _ = sgc_precompute(features[idx_train], train_adj, args.degree)
else:
    train_features = processed_features[idx_train]

test_features = processed_features[idx_test if args.test else idx_val]

train_features = train_features.cpu().detach().numpy()
test_features = test_features.cpu().detach().numpy()
train_labels = labels[idx_train].cpu().detach().numpy()
test_labels = labels[idx_test if args.test else idx_val].cpu().detach().numpy()
print("Finished Data Preprocessing")

# SVM
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X = scaler.fit_transfrom(X)
clf = SVC(kernel='linear')
clf.fit(train_features, train_labels, verbose=True)
import pdb; pdb.set_trace()

# xgboost
from xgboost import XGBClassifier
param = {}
param['objective'] = 'multi:softmax'
param['eval_metric'] = 'mlogloss'
param['eta'] = 0.1
param['max_depth'] = 10
param['nthread'] = 16
param['num_class'] = nclass
print("Start fitting")
dtrain = xgb.DMatrix(train_features, label=train_labels)
dtest = xgb.DMatrix(test_features, label=test_labels)
watchlist = [(dtrain, 'train'), (dtest, 'test')]
clf = xgb.train(param, dtrain, 50, watchlist, verbose_eval=1000)
test_pred = clf.predict(dtest)
test_acc = accuracy_score(test_labels, test_pred)
print("{} acc: {}".format("Test" if args.test else "Validation", test_acc))
