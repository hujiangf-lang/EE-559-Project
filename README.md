# Credit Card Fraud Detection under Highly Imbalanced Data

## Overview
This project focuses on credit card fraud detection under highly imbalanced data.  
We compare three machine learning models:

- Logistic Regression  
- Support Vector Machine (SVM)  
- Random Forest  

We also evaluate the impact of imbalance handling techniques such as class weighting and SMOTE.

---

## Dataset

Download the dataset from Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

After downloading, place the file in:

data/creditcard.csv

- Total samples: 284,807  
- Fraud cases: 492 (~0.17%)  

---

## Requirements

Python 3.7+

Install dependencies using:

pip install -r requirements.txt

---

## How to Run

Run each model separately:

python Logistic.py  
python SVM.py  
python Random_Forest.py  

---

## Project Structure

Project/  
├── data/  
│   └── creditcard.csv  
├── Logistic.py  
├── SVM.py  
├── Random_Forest.py  
├── requirements.txt  
└── README.md  

---

## Notes

- The dataset is highly imbalanced.  
- SMOTE is applied only to the training set to avoid data leakage.  
- The classification threshold is optimized using the validation set to maximize F1-score.  

---

## Results Summary

- Random Forest achieves the best overall performance.  
- Imbalance handling methods improve recall but reduce precision.  
- PR-AUC is more informative than ROC-AUC for this task.  