#!/usr/bin/env python
# coding: utf-8

# In[40]:


from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt


# In[26]:


irisData = load_iris()
x= irisData.data
print(x[:10])
print ('shape \n' , x.shape)
print('features names \n' , irisData.feature_names)


# In[28]:


y = irisData.target
print(y[:150])
print('features names \n' , irisData.target_names)


# In[30]:


digitsData = load_digits()


# In[34]:


a = digitsData.data
print(a[:2])
print ('shape \n' , a.shape)
#print('features names \n' , digitsData.feature_names)


# In[33]:


y = digitsData.target
print(y[:150])
print('features names \n' , digitsData.target_names)


# In[39]:


plt.gray()


# In[41]:


bd = load_breast_cancer()


# In[43]:


z = bd.data
print(z[:2])
print ('shape \n' , z.shape)
#print('features names \n' , bd.feature_names)


# In[44]:


dd = bd.target
print(dd[:150])
print('features names \n' , bd.target_names)


# In[ ]:




