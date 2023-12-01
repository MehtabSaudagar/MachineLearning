#!/usr/bin/env python
# coding: utf-8

# ### To perform EDA on the dataset and find out the best accuracy is given by which algorithm

# In[ ]:


# Census Income Project


# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv("census-income_ (1).csv")


# In[3]:


df.head()


# In[4]:


df.columns


# ## REPLACE THE WHITE SPACES FROM THE COLUMNS NAMES

# In[5]:


df.columns = df.columns.str.replace(' ','')


# In[6]:


df.columns


# In[ ]:


df.columns = df.columns.str.replace("-",".")


# In[ ]:


df.columns


# In[ ]:


df.isnull().sum()


# In[ ]:


df.describe()


# In[ ]:


census_ed = df[['education']]


# In[ ]:


census_ed.head()


# In[ ]:


#) Extracting all the columns from “age” to “relationship” and store it in “census_seq”.

census_seq = df.iloc[:,0:8]

census_seq.head()


# In[ ]:


#Extract the column number “0”, “5”, “6” and store it in “census_col”.

census_col = df.iloc[:,[0,5,6]]


# In[ ]:


census_col.head()


# In[ ]:


df['workclass'].value_counts()


# In[ ]:


#) Extracting all the male employees who work in state-gov and store it in “male_gov”.

male_gov=df[(df['workclass']=='State-gov') & (df['sex']=='Male')]


# In[ ]:


#Extract all the 39 year olds who either have a bachelor's degree or who are native of the United States and store the result in “census_us”.
#We are writing a condition where age should be 39 and storing it in census. This condition is then taken and applied on the 
#next one to get the result.

census=df[(df['age']==39)]


# In[ ]:


census_us=census[(census['education']=='Bachelors') | (census['native.country']==' United-States')]
census_us.head()


# In[ ]:


#sample() helps us to get random rows 
# to select 200 random rows

census_200=df.sample(200)
census_200


# In[ ]:


#Getting the count of different levels of the “workclass” column.
#value_counts() helps us to get the total count of all elements present in a particular column.

df[['workclass']].value_counts()


# In[ ]:


#Here we are calculating the mean of capital gain on the bases of work class
# calculate the mean of capital.gain grouped according to workclass

df.groupby(['workclass'])['capital.gain'].mean()


# In[ ]:


income=df["Unnamed:14"]
df["income"]=income
df.head()


# In[ ]:


df['income'].value_counts()


# In[ ]:


df=pd.DataFrame(df)
df.drop("Unnamed:14",axis=1,inplace=True)
df.head()


# In[ ]:


# Calculating the percentage of married people in the census data.

df['marital.status'].value_counts()


# In[ ]:


#calculating % of married people

total=len(df['marital.status'])
married=14976+418+23
precen_mar=(married/total)*100
precen_mar


# In[ ]:


# Calculatig the percentage of high school graduates earning more than 50,000 annually

df['education'].value_counts()


# # LOGISTIC REGRESSION

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[ ]:


lo=LogisticRegression()


# In[ ]:


df[['occupation']].value_counts()


# In[ ]:


# occupation is indpendent

x=df['occupation']
x=pd.DataFrame(x)
x


# In[ ]:


# giving particular labels to the elements in my columns

L=LabelEncoder()


# In[ ]:


x=L.fit_transform(x)


# In[ ]:


x = pd.DataFrame(x)


# In[ ]:


x.head()


# In[ ]:


y=df['income'].replace(' <=50K',0).replace(' >50K',1)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.35,random_state=42)


# In[ ]:


lo.fit(x_train,y_train) 


# In[ ]:


y_pred=lo.predict(x_test)


# In[ ]:


accuracy_score(y_test,y_pred)


# # DT

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


df['education'].value_counts()


# In[ ]:


df.education = df.education.replace(['Preschool','1st-4th','5th-6th','7th-8th','9th','10th','11th','12th'],'School')


# In[ ]:


df.education = df.education.replace(['Assoc-voc', 'Assoc-acdm', 'Prof-school', 'Some-college'], 'Higher')
df['marital.status']= df['marital.status'].replace(['Married-civ-spouse', 'Married-AF-spouse'], 'married')
df['marital.status']= df['marital.status'].replace(['Never-married'], 'not-married')
df['marital.status']= df['marital.status'].replace(['Divorced', 'Separated','Widowed','Married-spouse-absent'], 'other')
df.income = df.income.replace('<=50K', 0)
df.income = df.income.replace('>50K', 1)


# In[ ]:


df = df.apply(L.fit_transform)


# In[ ]:


df.head()


# In[ ]:


x=df.iloc[:,:-1]
y=df.iloc[:,-1]


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30,random_state=42)


# In[ ]:


dt=DecisionTreeClassifier()


# In[ ]:


dt.fit(x_train,y_train) 


# In[ ]:


y_pred=dt.predict(x_test)


# In[ ]:


accuracy_score(y_test,y_pred)


# # RANDOM FOREST

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=42)


# In[ ]:


rf=RandomForestClassifier()


# In[ ]:


rf.fit(x_train,y_train)


# In[ ]:


y_pred=rf.predict(x_test)


# In[ ]:


accuracy_score(y_test,y_pred)


# ### conclusions
# we have discussed all the classification models
# in which we have got the random forest as the best classifier
# because the accuracy is like in ascending order log_reg<des_tree<rand_forst
