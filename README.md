<h2 align="center">
  Raw ML Classifiers
</h2>

<p align="center">
  <a href="https://github.com/arkilpatel/SVAMP/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green">
  </a>
</p>


This repository contains implementations of six Machine Learning Classifiers using only standard Python libraries such as NumPy and Pandas.

<h2 align="center">
  <img align="center"  src="./images/Table1.png" alt="..." width="350">
</h2>

#### Dependencies

- compatible with python 3.6

#### Models

The current repository includes 6 implementations of Models:

- Naïve Bayes Classifier at `./naive_bayes.py`
- Decision Tree at `./decision_tree.py`
- Logistic Regression at `./logistic_regression.py`
- Random Forest at `./random_forest.py`
- Adaboost at `./adaboost.py`
- Stacking at `./stacking.py`

#### Datasets

The following datasets were used for evaluation:

- `Iris`
  - [Paper](https://www.aclweb.org/anthology/N16-1136.pdf) and [Github](https://github.com/sroy9/mawps).
  - `Data Size:` 2373 MWPs.
  - Evaluated by Cross-Validation over 5 splits.
  
- `Titanic`
  - [Paper](https://www.aclweb.org/anthology/2020.acl-main.92.pdf) and [Github](https://github.com/chaochun/nlu-asdiv-dataset).
  - `Data Size:` 1218
  - Evaluated by Cross-Validation over 5 splits.
  

#### Usage:

Here, we illustrate running the experiment 

##### Running Logistic Regression Model on 

If the folders for the 5 folds are kept as subdirectories inside the directory `../data/cv_asdiv-a:` (for eg, fold0 directory will have `../data/cv_asdiv-a/fold0/train.csv` and `../data/cv_asdiv-a/fold0/dev.csv`),

then, at `SVAMP/code/rnn_seq2seq:`

```shell
$	python -m src.main -mode train -gpu 0 -embedding roberta -emb_name roberta-base -emb1_size 768 -hidden_size 256 -depth 2 -lr 0.0002 -emb_lr 8e-6 -batch_size 4 -epochs 50 -dataset cv_asdiv-a -full_cv -run_name run_cv_asdiv-a
```



## Documentation:

#### Naïve Bayes Classifier:

```python
class NaiveBayes(type = “Gaussian”, prior = None)
```

This class implements Naïve Bayes with Gaussian and Multinomial types.

<table>
<tr>
    <td>Parameters</td>
    <td><b>type:</b> str, ‘Gaussian’ or ‘Multinomial’, (default= ‘Gaussian’) <br> Specifies which type ‘Gaussian’ or ‘Multinomial’ is to be used. <br> <br>
<b>prior:</b> array-like, shape (n_classes,) <br> Prior probabilities of the classes. If specified the priors are not adjusted according to the data.
</td>
</tr>
<tr>
    <td>Attributes</td>
    <td><b>class_log_prior:</b> array, shape(n_classes) <br> Prior probability of each class._
</td>
</tr>
</table>



For any clarification, comments, or suggestions please contact [Arkil](http://arkilpatel.github.io/).