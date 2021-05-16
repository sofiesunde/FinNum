# Understanding Numerals in Financial Social Media
### Project for TDT4310 - Intelligent Text Analytics and Language Understanding

The goal for this project is to understand fine-grained numeral information in social media data. This project focuses on classifying financial numerals from tweets into seven different categories with machine learning classifiers. The original crowdsourced task is the _NTCIR-14 FinNum_ task from the _Department of Computer Science and Information Engineering_ at National Taiwan University, given in 2018. This task can be found at [NTCIR-14 FinNum: Fine-Grained Numeral Understanding in Financial Tweets](https://sites.google.com/nlg.csie.ntu.edu.tw/finnum)  
To understand the Numeral Taxonomy used in the task and the background for the project, see [Chung-Chi Chen, Hen-Hsen Huang, Yow-Ting Shiue, Hsin-Hsi Chen: Numeral Understanding in Financial Tweets for Fine-grained Crowd-based Forecasting](http://nlg.csie.ntu.edu.tw/~cjchen/papers/Numeral_Understanding_WI.pdf) 


## Datasets

The datasets used in this project can be accessed at [NTCIR-14 FinNum: Data](https://sites.google.com/nlg.csie.ntu.edu.tw/finnum/data) with the Stocktwits developer API.  
To run the given rebuild program, follow the _README of rebuild dataset_. The _HTTP library Requests_ can be installed with [pip](https://pypi.org/project/pip/):

```python
pip install requests 
```

The training set contains 3949 instances.  
The development set contains 449 instances.  
The test set contains 763 instances. 

The traning dataset includes these features:  
idx  
id  
target_num  
category  
subcategory  
tweet 

Where category (and eventually subcategory) is the output of the machine learing models. 

## Prerequisites

```python
pip install json
pip install sklearn
pip install nltk
pip install pandas
pip install matplotlib
```

## Architecture 

![FinNumArchitecture](https://user-images.githubusercontent.com/74187128/118411071-24df6500-b693-11eb-8bfe-16cd76b999b1.png)

```python preprocessing/``` contains code for reading and preprocsessing the data from the datasets. The models are written in ```python main.py``` where the whole system is managed. ```python main.py``` is not used as of now, but provides a great structure to further expand the system. 

## Results

The Random Forest Classifier performs better than the Support Vector Machine Classifiers, however their performances are pretty low and some future adjustments will have to be made. 

## Roadmap

* POS Tagging of Tweets
* Considering Changing Chosen Classifiers
* Further Implementation of Traditional NLP Methods
