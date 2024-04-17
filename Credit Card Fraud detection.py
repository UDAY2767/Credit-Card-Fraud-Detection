#!/usr/bin/env python
# coding: utf-8

# # CREDIT CARD FRAUD DETECTION  USING MACHINE LEARNING 

# # Context
# It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.
# 
# Content
# 
# The dataset contains transactions made by credit cards in September 2013 by European cardholders.
# This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
# 
# It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

# # IMPORT REQUIRED LIBRARIES  

# In[2]:


import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt


# # MODELLING

# In[5]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import IsolationForest


# # METRICS 

# In[6]:


from sklearn.metrics import precision_score, recall_score, make_scorer


# # IMPORTING THE DATASET 
# * Only the first 80,000 rows

# In[11]:


file_path = r"C:\Users\udayu\Downloads\creditcard.csv\creditcard.csv"
df = pd.read_csv(file_path)[:80000]


# In[12]:


df.head()


# In[13]:


df.tail()


# # PREPROCESSING

# # Features
# * Extracting the feature into X and Classes into Y

# In[16]:


X = df.drop(columns=['Time', 'Amount', 'Class']).values
y = df['Class'].values

X


# # Target labels
# 1 denotes the fraud cases   And
# 0 denotes the genuine cases

# In[17]:


classes, classes_count = np.unique(df['Class'].values, return_counts=True)

plt.figure(figsize=(8, 6))
bars = plt.bar(classes, classes_count, color=['blue', 'red'])

for bar, count in zip(bars, classes_count):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20, count,
             ha='center', va='bottom', color='black')

plt.title("Target labels")
plt.xticks(classes)

plt.xlabel("Classes")
plt.ylabel("Cases")
plt.legend(bars, ["Non-fraud", "Fraud"])

plt.show()


# # MODELLING

# # Logistic Regression
# Using LogisticRegression for binary classification

# In[18]:


model = LogisticRegression(class_weight={0: 1, 1: 2}, max_iter=1000)

pred1 = model.fit(X, y).predict(X).sum()

print(f"Predicted number of Fraud cases : {pred1}")


# # Grid Search
# * Using GridSearchCV to determine the optimal class weights
# * 4-fold Cross-Validation is being used

# In[19]:


grid = GridSearchCV(
    estimator=LogisticRegression(max_iter=1000),
    param_grid={'class_weight': [{0: 1, 1: v} for v in range(1, 4)]},
    cv=4,
    n_jobs=-1)

grid


# # Training The Grid

# In[20]:


grid.fit(X,y)


# In[21]:


#Results of the cross-validation
#Finding out the optimal class weights


# In[23]:


cv_results = pd.DataFrame(grid.cv_results_)

cv_results.sort_values(by='mean_test_score', ascending=False).head(5)


# # METRICS

# # Precision and Recall
# * Using different scoring metrics

# In[24]:


p_score = precision_score(y, grid.predict(X))
r_score = recall_score(y, grid.predict(X))

print(f"Precision Score : {p_score}\nRecall Score    : {r_score}")


# Adding precision and recall to Grid Search
# * Optimizing the precision score

# In[25]:


grid_pr = GridSearchCV(
    estimator=LogisticRegression(max_iter=1000),
    param_grid={'class_weight': [{0: 1, 1: v} for v in np.linspace(1, 20, 30)]},
    scoring={"precision": make_scorer(precision_score), "recall_score": make_scorer(recall_score)},
    refit='precision',
    return_train_score=True,
    cv=10,
    n_jobs=-1)

grid_pr


# In[28]:


#Training the grid_pr 
grid_pr.fit(X,y)


# # Results of the cross-validation
# Finding out the optimal class weights based on precision score

# In[29]:


cv_results_pr = pd.DataFrame(grid_pr.cv_results_)

cv_results_pr.head()


# In[31]:


# Best mean precision_score
cv_results_pr.sort_values(by='mean_test_precision', ascending=False).head(1)


# # Plotting the performance metrics
# * We observe the difference in performance of the model on training vs. on the testing data
# * The intersection b/w Recall and Precision occurs much earlier in training data

# Test Data

# In[32]:


# Test Data
plt.figure(figsize=(12, 4))

for score in ['mean_test_recall_score', 'mean_test_precision']:
    plt.plot([_[1] for _ in cv_results_pr['param_class_weight']], cv_results_pr[score], label=score)

plt.title("Class Weights vs. Testing Scores")
plt.xlabel("Class Weight")
plt.ylabel("Score")

plt.legend(["Mean Recall", "Mean Precision"])

plt.show()


# Train Data 

# In[34]:


# Train Data
plt.figure(figsize=(12, 4))

for score in ['mean_train_recall_score', 'mean_train_precision']:
    plt.plot([_[1] for _ in cv_results_pr['param_class_weight']], cv_results_pr[score], label=score)

plt.title("Class Weights vs. Training Scores")
plt.xlabel("Class Weight")
plt.ylabel("Score")

plt.legend(["Mean Recall", "Mean Precision"])

plt.show()


# # New performance metric
# * We will create a new metric that optimizes on the min() of precision and recall scores

# In[35]:


def min_precision_recall1(y_true, y_pred):
    p_score = precision_score(y_true, y_pred)
    r_score = recall_score(y_true, y_pred)
    
    return min(p_score, r_score)


# # Creating and training the new Grid
# * This makes use of the new metric
# * Also, transactions are weighted by their Amount

# In[36]:


grid_pr2 = GridSearchCV(
    estimator=LogisticRegression(max_iter=1000),
    param_grid={'class_weight': [{0: 1, 1: v} for v in np.linspace(1, 20, 30)]},
    scoring={"precision": make_scorer(precision_score), 
             "recall_score": make_scorer(recall_score),
             "min_recall_precision": make_scorer(min_precision_recall1)},
    refit='min_recall_precision',
    return_train_score=True,
    cv=10,
    n_jobs=-1)

grid_pr2.fit(X, y)


# Result of the cross-validation

# In[37]:


cv_results_pr2 = pd.DataFrame(grid_pr2.cv_results_)

cv_results_pr2.head()


# In[38]:


# Best of the new score
cv_results_pr2.sort_values(by='mean_test_min_recall_precision', ascending=False).head(1)


# # Plotting the performence metrics

# without the weighted Amount

# In[39]:


plt.figure(figsize=(12, 4))

for score in ['mean_test_recall_score', 'mean_test_precision', 'mean_test_min_recall_precision']:
    plt.plot([_[1] for _ in cv_results_pr2['param_class_weight']], cv_results_pr2[score], label=score)

plt.title("Class Weights vs. Training Scores")
plt.xlabel("Class Weight")
plt.ylabel("Score")

plt.legend(["Mean Recall", "Mean Precision", "Minimum of Recall and Precision"])

plt.show()


# with the weighted Amount

# In[40]:


def min_precision_recall2(est, X, y_true, sample_weight=None):
    y_pred = est.predict(X)
    p_score = precision_score(y_true, y_pred)
    r_score = recall_score(y_true, y_pred)
    
    return min(p_score, r_score)

grid_pr2_2 = GridSearchCV(
    estimator=LogisticRegression(max_iter=1000),
    param_grid={'class_weight': [{0: 1, 1: v} for v in np.linspace(1, 20, 30)]},
    scoring={"precision": make_scorer(precision_score), 
             "recall_score": make_scorer(recall_score),
             "min_recall_precision": min_precision_recall2},
    refit='min_recall_precision',
    return_train_score=True,
    cv=10,
    n_jobs=-1)

grid_pr2_2.fit(X, y, sample_weight=np.log(1 + df['Amount']))

cv_results_pr2_2 = pd.DataFrame(grid_pr2_2.cv_results_)


# In[41]:


plt.figure(figsize=(12, 4))

for score in ['mean_test_recall_score', 'mean_test_precision', 'mean_test_min_recall_precision']:
    plt.plot([_[1] for _ in cv_results_pr2_2['param_class_weight']], cv_results_pr2_2[score], label=score)

plt.title("Class Weights vs. Training Scores")
plt.xlabel("Class Weight")
plt.ylabel("Score")

plt.legend(["Mean Recall", "Mean Precision", "Minimum of Recall and Precision"])

plt.show()


# # Treating fraud cases as outliers
# Using the IsolationForest model

# In[42]:


model2 = IsolationForest().fit(X)

print(f"Non-outliers\t: {Counter(np.where(model2.predict(X) == -1, 1, 0))[0]}\nOutliers\t: {Counter(np.where(model2.predict(X) == -1, 1, 0))[1]}")


# # Defining a grid for IsolationForest
# New scoring metrics i.e. we treat fraud cases as outliers

# In[43]:


def outlier_precision(mod, X, y):
    pred = mod.predict(X)
    return precision_score(y, np.where(pred==-1, 1, 0))

def outlier_recall(mod, X, y):
    pred = mod.predict(X)
    return recall_score(y, np.where(pred==-1, 1, 0))

grid_o = GridSearchCV(
    estimator=IsolationForest(),
    param_grid={
        'contamination': np.linspace(0.001, 0.02, 10)
    },
    scoring={
        'precision': outlier_precision,
        'recall': outlier_recall,
    },
    refit='precision',
    cv=5,
    n_jobs=-1
)

grid_o.fit(X, y)


# results of the cross-validation

# In[44]:


cv_results_o = pd.DataFrame(grid_o.cv_results_)

cv_results_o.sort_values(by='mean_test_precision', ascending=False).head()


# In[45]:


#plotting the performence metrics
plt.figure(figsize=(12, 4))

for score in ['mean_test_recall', 'mean_test_precision']:
    plt.plot([_ for _ in cv_results_o['param_contamination']], cv_results_o[score], label=score)

plt.title("Contamination vs. Training Scores")
plt.xlabel("Contamination")
plt.ylabel("Score")

plt.legend(["Mean Recall", "Mean Precision"])

plt.show()


# # Conclusion
# Well, the IsolationForest model is of no use. We should stick to Regression

# In[ ]:




