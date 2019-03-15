## Simplifying Graph Convolutional Networks
#### Authors: 
* [Felix Wu](https://scholar.google.com.tw/citations?user=sNL8SSoAAAAJ&hl=en)*
* [Tianyi Zhang](https://scholar.google.com/citations?user=OI0HSa0AAAAJ&hl=en)*
* [Amauri Holanda de Souza Jr.](https://scholar.google.com/citations?hl=en&user=lP0LBI4AAAAJ&view_op=list_works&sortby=pubdate)*
* Christopher Fifty
* [Tao Yu](http://jhc.sjtu.edu.cn/public/home/taoyu/)
* [Kilian Q. Weinberger](http://kilian.cs.cornell.edu/index.html)

*: Equal Contribution

### Overview
In good faith of facilitating future research, this branch contains additional
codebase of our ablation studies. Notably, in `normalization.py` we provide more
functions on normalizing the adjacency matrix. These design choices corresponds
to different fixed spectral filters.

Specifically, we can design a low pass filter (`A' = I - 1/2 * D^-1/2 * A *
D^-1/2`). This low pass filter performs competitively, in particular on the
Reddit dataset. Such a result corresponds well with our spectral analysis in the paper. 
For reference: 

Dataset | Metric | Filter
:------:|:------:|:-----------:|
Cora    | Acc: 80.4 %     | Low-Pass, degree=2
Citeseer| Acc: 71.9 %     | Low-Pass, degree=2
Pubmed  | Acc: 78.5 %     | Low-Pass, degree=2
Reddit  | F1:  *95.4* %     | Low-Pass, degree=4

This branch is still in progress and may change in the future. New results based
on this branch will be included in the supplementary materials of the next version of our paper. 

if you find this repo (branch) useful, please cite: 
```
@article{sgc,
  title={Simplifying Graph Convolutional Networks},
  author={Wu, Felix and Zhang, Tianyi and Souza Jr., Amauri Holanda and Fifty, Christopher and Yu, Tao and Weinberger, Kilian Q.},
  journal={arXiv preprint arXiv:1902.07153},
  year={2019}
}
```

### Usage
See possibile normalization functions by `python citation.py --help`. Example commands include: 

```
$ python citation.py --dataset cora --normalization AugRWalk --degree 2
$ python citation.py --dataset citeseer --normalization AugNormAdj --concat
$ python citation.py --dataset pubmed --normalization LowPass --degree 2
```

Reddit:
```
$ python reddit.py --inductive --test --normalization LowPass --degree 4
```
