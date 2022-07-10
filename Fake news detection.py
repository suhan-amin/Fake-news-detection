#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


import nltk
nltk.download('stopwords')


# In[3]:


print(stopwords.words('english'))


# ### Data preprocessing        [1->Fake News and 0->Real News]

# In[4]:


news_dataset=pd.read_csv('train.csv')
news_dataset.shape


# In[5]:


news_dataset.head()


# In[6]:


#missing values in the dataset
news_dataset.isnull().sum()


# In[7]:


#replacing null values with empty strings
news_dataset=news_dataset.fillna('')


# In[8]:


#merging author name and title
news_dataset['content']=news_dataset['author']+' '+news_dataset['title']
print(news_dataset['content'])


# In[9]:


X=news_dataset.drop(columns='label',axis=1)
Y=news_dataset['label']
print(X)
print(Y)


# ### Stemming

# In[10]:


port_stem=PorterStemmer()
def stemming(content):
    
    #to remove everything(such as nummbers etc) except alphabets from content and replace them with empty spaces
    stemmed_content=re.sub('[^a-zA-Z]',' ',content)
                           
    stemmed_content=stemmed_content.lower()
    stemmed_content=stemmed_content.split()
    stemmed_content=[port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content=' '.join(stemmed_content)
    return stemmed_content


# In[11]:


news_dataset['content']=news_dataset['content'].apply(stemming)


# In[12]:


print(news_dataset['content'])


# In[13]:


X=news_dataset['content'].values
Y=news_dataset['label'].values


# In[14]:


print(X)


# In[15]:


print(Y)


# ### Converting text to numerical data

# In[16]:


vectorizer=TfidfVectorizer()
vectorizer.fit(X)
X=vectorizer.transform(X)
print(X)


# ### Training and Testing data 

# In[17]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)


# ### Training model using Logistic Regression

# In[18]:


model=LogisticRegression()


# In[19]:


model.fit(X_train,Y_train)


# ### Accuracy Score 

# In[20]:


X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
print('Accuracy Score of the training data:',training_data_accuracy)


# In[21]:


X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print('Accuracy Score of the test data:',test_data_accuracy)


# ### Prediction
# 

# In[57]:


X_new=X_test[0]
prediction=model.predict(X_new)
print(prediction)

if(prediction[0]==0):
    print('The news is Real')
else:
    print('The news is Fake')


# In[58]:


print(Y_test[0])


# In[96]:


X_new=X_test[1]
prediction=model.predict(X_new)
print(prediction)

if(prediction[0]==0):
    print('The news is Real')
else:
    print('The news is Fake')


# In[97]:


print(Y_test[1])


# ### Predicting using News as input
# 
# 

# In[102]:


def output_label(n):
    if n==1:
        return 'Fake'
    elif n==0:
        return 'Real'
    
def manual_testing(news):
    testing_news={'text':[news]}
    new_def_test=pd.DataFrame(testing_news)
    new_def_test['text']=new_def_test['text'].apply(stemming)
    new_x_test=new_def_test['text']
    new_xv_test=vectorizer.transform(new_x_test)
    pred_LR=model.predict(new_xv_test)
    
    return print("\nThe news is: {}".format(output_label(pred_LR)))


# In[108]:


news=str(input())
manual_testing(news)


# In[ ]:





# In[ ]:




