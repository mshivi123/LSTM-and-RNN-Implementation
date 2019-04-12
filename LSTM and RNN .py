#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import matplotlib.pyplot as plt


# In[19]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


# In[20]:


data = [[i for i in range(100)]]


# In[21]:


data = np.array(data, dtype=float)
target = [[i for i in range(1,101)]]
target = np.array(target, dtype=float)


# In[22]:


data = data.reshape((1, 1, 100)) 
target = target.reshape((1, 1, 100))


# In[23]:


x_test=[i for i in range(100,200)]
x_test=np.array(x_test).reshape((1,1,100));


# In[24]:


y_test=[i for i in range(101,201)]
y_test=np.array(y_test).reshape(1,1,100)


# In[25]:


model = Sequential() 


# In[26]:


model.add(LSTM(100, input_shape=(1, 100),return_sequences=True))


# In[27]:


model.add(Dense(100))


# In[28]:


model.compile(loss='mean_absolute_error', optimizer='adam',metrics=['accuracy'])


# In[29]:


model.fit(data, target, nb_epoch=10000, batch_size=1, verbose=2,validation_data=(x_test, y_test))


# In[33]:


results = model.predict(data)


# In[34]:


plt.scatter(range(20),results,c='r')
plt.scatter(range(20),y_test,c='g')
plt.show()


# In[ ]:




