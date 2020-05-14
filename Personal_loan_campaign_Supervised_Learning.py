#!/usr/bin/env python
# coding: utf-8

# In[1]:


### ENABLES GRAPGH PLOTTING IN JUPYTER
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


### LIBRARIES
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import metrics
import seaborn as sns

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


# In[3]:


### READING CSV FILE
df = pd.read_csv("Bank_Personal_Loan_Modelling.csv")
df


# In[4]:


### DATA TYPES
df.dtypes


# In[5]:


# 1 FLOAT ATTRIBUTE: CCAVG
# 13 INTEGERS


# In[6]:


df.describe()


# In[7]:


# EXPERIENCE CONTAINS NEGATIVE NUMBERS


# In[8]:


### CONVERT EXPERIENCE ATTRIBUTE TO CONTAIN NON-NEGATIVE NUMBERS
### USE .abs FUNCTION

df['Experience'] = df['Experience'].abs()

df.describe().transpose()


# In[9]:


### SEABORN LIBRARY IMPORT: STATISTICAL PLOTS
### COMPARING ALL ATTRIBUTES
import seaborn as sns
df_attr = df.iloc[:,0:12] 

sns.pairplot(df_attr)


# In[10]:


# STRONG POSITIVE RELATIONSHIP: AGE & EXPERIENCE
# NO LINEAR RELATIONSHIP BETWEEN AGE & INCOME
# NO LINEAR RELATIONSHIP BETWEEN AGE & EXPERIENCE


# In[11]:


df.dtypes


# In[12]:


df.groupby(["Personal Loan"]).count() 


# In[13]:


# DATA IS SKEWED IN TERMS OF TARGET COLUMN
# VERY FEW RECORDS OF PEOPLE WHO PREVIOUSLY TOOK OUT PERSONAL LOANS


# In[15]:


### SEPERATION OF INDEPENDENT ATTRIBUTES: STORE THEM IN X-ARRAY
### STORE TARGET COLUMN IN Y-ARRAY 

X_df = df.loc[:, df.columns != 'Personal Loan']
y_df = df.loc[:, df.columns == 'Personal Loan']


# In[16]:


#### MODEL: LOGISTIC


# In[17]:


### TRAINING & TEST DATA: 60:40
### DATA PREPARATION FOR LOGISTIC REGRESSION

features=X_df.iloc[:, 0:10]
features_array = features.values   
target_labels = y_df.values


test_size = 0.40 

### RANDOM NUMBER SEEDING: REPEATABILITY OF CODE WHEN USING RANDOM FUNCTIONS
seed = 7  

X_train, X_test, y_train, y_test = model_selection.train_test_split(features_array, target_labels, test_size=test_size, random_state=seed)

### CONVERT 1 D VECTOR INTO 1 D ARRAY
y_train = np.ravel(y_train)   


# In[18]:


### LOGISTIC REGRESSION TP PREDICT PERSONAL LOAN AFFINITY
### REMOVED NUMERIC BINNED COLUMNS  

model = LogisticRegression()
model.fit(X_train, y_train)
model_score = model.score(X_test, y_test)
y_predict = model.predict(X_test)
print(model_score)
print(metrics.confusion_matrix(y_test, y_predict))


# In[19]:


# ACCURACY SCORE OF 0.912 
# BUT ACCURACY SCORE IS AT MODEL LEVEL, WHICH MAKES IT UNRELIABLE


# In[20]:


### MODEL FIT SUMMARY
print(metrics.classification_report(y_test, y_predict))


# In[21]:


# LOW PRECISION RECALL FOR CLASS 1


# In[22]:


#### MODEL: NAIVE BIAS


# In[23]:


### TRAIN AND TEST DATA SET
### DATA PREP

features=X_df.iloc[:, 0:10]

target_labels = df.loc[:, df.columns == 'Personal Loan']


X_array = features.values
y_array = target_labels.values


test_size = 0.40 
seed = 7  

X_train, X_test, y_train, y_test = model_selection.train_test_split(X_array, y_array, test_size=test_size, random_state=seed)
y_train = np.ravel(y_train)   


# In[24]:


### INVOKING NB GAUSSIAN FUNCTION 
### FITTING MODEL IN TRAINING DATA SET

model = GaussianNB()
model.fit(X_train, y_train)

predictions=model.predict(X_test)

### ACCURACY TEST OF MODEL
print(metrics.confusion_matrix(y_test,predictions))


# In[25]:


### PREDICTIONS
expected = y_test
predicted = model.predict(X_test)

# MODEL FIT SUMMARY
print(metrics.classification_report(expected, predicted))


# In[26]:


# CLASS 1 METRICS: NOT IN ACCEPTABLE RANGE (80% & ABOVE)


# In[27]:


#### MODEL: KNN


# In[29]:


from sklearn.neighbors import KNeighborsClassifier
NNH = KNeighborsClassifier(n_neighbors= 3 , weights = 'distance')
NNH.fit(X_train, y_train)


# In[30]:


predicted_labels = NNH.predict(X_test)


# In[31]:


print(metrics.confusion_matrix(y_test, predicted_labels))


# In[32]:


### MODEL FIT SUMMARY
print(metrics.classification_report(y_test, predicted_labels))


# In[33]:


# RECALL FOR CLASS ONE IS THE LEAST


# In[34]:


### SCALING: Z-SCORE 
from sklearn import preprocessing
X_train_scaled = preprocessing.scale(X_train)
X_test_scaled = preprocessing.scale(X_test)
NNH.fit(X_train_scaled, y_train)


# In[35]:


predicted_labels = NNH.predict(X_test_scaled)


# In[36]:


print(metrics.confusion_matrix(y_test, predicted_labels))


# In[37]:


### MODEL FIT SUMMARY
print(metrics.classification_report(y_test, predicted_labels))


# In[ ]:


#SCALED KNN HAS PROVIDED THE BEST RESULT 

