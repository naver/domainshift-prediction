<img width="600" alt="NLE" src="https://i.imgur.com/488nYbr.jpg">  <img display="inline" width="100" alt="EMNLP" src="https://i.imgur.com/8c0QJBF.jpg">
### To Annotate or Not:question: Predicting Performance Drop under Domain Shift :mag_right:

- EMNLP2019 Paper: [https://www.aclweb.org/anthology/D19-1222/](https://www.aclweb.org/anthology/D19-1222/)
- Blog post: [http://bit.ly/hadyelsahar-emnlp2019-blog1](http://bit.ly/hadyelsahar-emnlp2019-blog1)


#### Authors:

Hady Elsahar: hady.elsahar@naverlabs.com \
Matthias Galle: matthias.galle@naverlas.com

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons Licence" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a> This repo is mainly distributed under the CC BY-NC-SA 4.0 License. See LICENSE-CC-BY-NC-SA-4.0.md for more information.
The modified AllenNLP modules are distributed under the Apache License, Version 2.0. See LICENSE-APACHE-2.0.txt for more information. For detailed information please refer to LICENSE.txt and NOTICE.txt files.

-----------

## :pill: tldr;
We propose a method that is able to predict the drop in accuracy of a trained model. Our method can predict the drop in accuracy with an error rate as little as 2.15% for sentiment analysis and 0.89% for POS tagging respectively, without needing any labeled examples from the target domain, only assuming the availability of small fixed number of labeled evaluation datasets from several other domains. 

## :small_red_triangle_down: Brief

* In this paper we study the problem of predicting the performance drop of modern NLP models due to domain-shift, in the absence of any target domain labels. 

* We investigate three families of methods (H-divergence, Reverse Classification Accuracy and Confidence measures), show how they can be used to predict the performance drop and study their robustness to adversarial domain-shifts.

* Afterwards we employ those metrics to estimate the performance drop on a new target domain Dt by regressing on those metrics and their associated real performance drop. 

<p align="center"> <img width="400" alt="Approach overview" src="https://i.imgur.com/pmKzp4W.png"> </p>


# Dataset
### Sentiment Analysis

Dataset home page: http://snap.stanford.edu/data/web-Amazon-links.html  

Summary of Dataset preprocessing:

- Download all `all.txt.gz` and convert to csv table
- Inner megre with `categories.txt.gz` on product id

For each product in `categories.txt.gz` there are several categories
```
B0000012DK
 Music, Blues
 Music, Pop
```
We pick the first one to represent the category. 
​
- Group by categories 
- Randomly sample 31k reviews per each category 
  - Train1 10k
  - Train2 10k (for domain discrimination calculation of PAD metric)
  - Valid 10k
  - Test 1k 


### POS Tagging

Dataset home page: [Universal Dependencies datasets](https://universaldependencies.org/) 

Download 4 English UD datasets `ParTUT, GUM, EWT, LinES`:
```
for SPLIT in train dev test
do
for DATASET in partut gum ewt lines
do
wget https://raw.githubusercontent.com/UniversalDependencies/UD_English-${DATASET}/master/en_${DATASET}-ud-${SPLIT}.conllu
done
done
```

We  split  the  EWTdataset (UD for English web treebank) accordingto each sub-category, while keeping the rest of thesmaller datasets as is.   
This yields in total 8 do-mains with roughly comparable sizes (∼4Ksen-tences each) yielding in total 56 domain-shift sce-narios.

# Code and Experiments

## Installation and Run Experiments

Load a new virtual environment and install requirements

```
virtualenv venv 
source ./venv/bin/activate
pip -r install requirements.txt
```

## Run Sentiment Analysis Domain-shift Experiment:
Run a sentiment analysis experiments for a model of h_d 256 and domain
classifier of hidden dim 256 initializd with glove embeddings. 
Output file should yield values of all metrics as well as drop 
in performance.

To replicate the experiments in the paper 
one should run a script to with all perumations of src and tgt domains with 
with several classifiers of different dimensions and initializations.
More details see [paper](https://www.aclweb.org/anthology/D19-1222.pdf) 
and [supplementary material](https://www.aclweb.org/anthology/attachments/D19-1222.Attachment.pdf)\
For more parameters see config in `run_sentiment.py`


```
python run_sentiment.py \
--d1train "./path/to/datasets/sentiment_analysis/srcdomain_train.tsv" \
--d1train2 "./path/to/datasets/sentiment_analysis/src-domain_train2.tsv" \
--d1valid "./path/to/datasets/sentiment_analysis/src-domain_valid.tsv" \
--d1test  "./path/to/datasets/sentiment_analysis/src-domain_test.tsv" \
--d2train "./path/to/datasets/sentiment_analysis/tgt-domain_train.tsv" \
--d2valid "./path/to/datasets/sentiment_analysis/tgt-domain_valid.tsv" \
--d2test  "./path/to/datasets/sentiment_analysis/tgt-domain_test.tsv" \
--DATASET_READER reviews --DD_DATASET_READER dd-reviews  \
--SEED 333 \
--HIDDEN_DIM  256\
--DD_HIDDEN_DIM  256\
--DROPOUT  0.5 \
--DD_DROPOUT 0.5 \
--N_LAYERS  10 \
--LR 0.01 \
--DD_LR 0.01 \
--GPU 0 \
--W_EMB glove \
--EPOCHS 30 --DD_EPOCHS 30 \
--out temp.log
```


## Run POS Tagging Domain-shift Experiment:
Run a pos tagging experiments for a model initialized with ELMO embeddings
Output file should yield values of all metrics as well as drop in performance.

To replicate the experiments in the paper 
one should run a script to with all perumations of src and tgt domains with 
with several Model Architecture of different dimensions and initializations.
More details see [paper](https://www.aclweb.org/anthology/D19-1222.pdf) 
and [supplementary material](https://www.aclweb.org/anthology/attachments/D19-1222.Attachment.pdf)\
For more parameters see config in `run_POS.py`

```
Python run_POS.py \
--d1train "./path/to/datasets/ud+/en_ParTUT-ud-train.conllu" \
--d1valid "./path/to/datasets/ud+/en_ParTUT-lines-dev.conllu" \
--d1test "./path/to/datasets/ud+/en_ParTUT-ud-dev.conllu" \
--d2train "./path/to/datasets/ud+/email_en_GUM-ud-train.conllu" \
--d2valid "./path/to/datasets/ud+/email_en_GUM-ud-dev.conllu" \
--d2test "./path/to/datasets/ud+/email_en_GUM-ud-dev.conllu" \
--DATASET_READER "custom_universal_dependencies"  \
--DD_DATASET_READER "dd_custom_universal_dependencies"  \
--GPU 0 \
--EPOCHS 5 \
--W_EMB ELMO \
--N_LAYERS 2 \
--DROPOUT 0.5 \
--DD_HIDDEN_DIM 80 \
--LR 0.01
```



