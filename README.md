# Relative Consistency


## Citation

[How Much Consistency Is Your Accuracy Worth?](https://arxiv.org/abs/2310.13781)


## Installation

Simply clone the repository:

```
git clone https://github.com/jacobkj314/relative-consistency
```

and then place the file relative_consistency.py where you need to use it

## Use

This measurement is intended for use with contrastive datasets such as [CondaQA](https://github.com/AbhilashaRavichander/CondaQA) or [ScoNe](https://github.com/selenashe/ScoNe/), from which you can construct pairs of related instances which vary in a feature, such as the presence, phrasing, or syntactic position of negation

After running 
```
from relative_consistency import relative_consistency
```
to import the function, simply run
```
relative_consistency(n,a,c)
```
where n is the total number of pairs, a is the number of accurate instances, and c is the number of cons


## Replicating CondaQA training/evaluation

This code is adapted from the code used to evaluate [CondaQA](https://github.com/AbhilashaRavichander/CondaQA), which I have been permitted to share here. Anything messy or counterintuitive in the code is my doing.

Before running, install anaconda if you haven't already:
```
wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh
bash Miniconda3-py37_4.12.0-Linux-x86_64.sh
```
Once you have installed conda, use it to create and set up the conda environment
```
conda create -n 38 python=3.8
conda activate 38
pip install -r requirements.txt
```

To set hyperparameters:
-edit the file `rep/src/Supervised/hparams.py` to set the weights of MLE and CE loss
-edit the "`make`" and "`model`" in the file `script.sh` to set which model you would like to train. By default, `unifiedqa-v2-t5-large-1251000` is trained. If you would like to train an existing local checkpoint, set this information in the `model` variable, and set `is_checkpoint=true` and `make=""`
-add `-use_test` after `python dataBundler.py` in the file `script.sh` to get the test-set results (instead of dev-set results)

To run:
```
cd rep/src/Supervised/
bash script.sh
```

When it has run, there will be a directory `results` in `rep/src/Supervised` containing `.txt` files of the results, which includes consistencies and accuracies to put into `relative-consistency.py`