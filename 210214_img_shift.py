#!/usr/bin/env python
# coding: utf-8

# In[5]:


import cv2


# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


from PIL import Image


# In[22]:


# img = Image.open('sample/IMG_20201208_220156.jpg')
# img = img.convert('L')
#img = img.resize((28,28))
img = cv2.imread('sample/IMG_20201208_220156.jpg',0)


# In[23]:


img_n = np.array(img)
plt.imshow(img_n,cmap='gray')


# In[ ]:


#copying images with parallel shift, rotation (and, magnification)
#shift 2dimensional, rotatoin: angle, magnification


# In[13]:


shift_pixel = 100


# In[19]:


#numpy のslicing
img1 = img_n[0:img.size[1]-shift_pixel,0:img.size[0]-shift_pixel]


# In[31]:


img.shape


# In[26]:


(h,w) = img.shape


# In[46]:


import math
r = math.pi/180*90
#m = np.float32([[math.cos(r), -math.sin(r), h], [math.sin(r), math.cos(r), w]])
m = np.float32([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0]])

## [[-1.0000000e+00 -1.2246469e-16  1.0520000e+03]
##  [ 1.2246469e-16 -1.0000000e+00  1.0240000e+03]]

im_transformed = cv2.warpAffine(img, m,(w,h))


# In[47]:


cv2.imshow('ig',im_transformed)


# In[43]:


im_transformed_n = np.array(im_transformed)


# In[44]:


plt.imshow(im_transformed_n,cmap='gray')


# In[ ]:




