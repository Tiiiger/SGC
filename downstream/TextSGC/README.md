# TextSGC

Implementation of SGC for text classification following
([Graph Convolutional Networks for Text Classification](https://arxiv.org/abs/1809.05679)).
Code modified from the original repo of
Text-GCN(https://github.com/yao8839836/text_gcn) and the pytorch implementation
of GCN (https://github.com/tkipf/pygcn).

## Data

We get the data from [original repo](https://github.com/yao8839836/text_gcn) and
process with `remove_words.py`, `build_graph.py`. We modify these two files from
the original repo and made some slight changes.
We share the processed data in a zip file
[online](https://drive.google.com/file/d/10kx3z3bjYFoeRjjg1_DZOAP39Jln0BCh/view?usp=sharing).
Please uncompress the file and put it under `data/`. If you want to redo the
preprocessing, see options with `python build_graph.py --help` and `python
remove_words.py --help`

## Usage

There is only one hyperparameter needs to be set and we tune it with hyperopt.
We provide the tuned hyperparameters under `tuned_result`.
You can replicate the paper reported result by
```
python --dataset <dt> --tuned
```

See `tuning.py` for the tuning details.
