#!/usr/bin/env python
# coding: utf-8

# In[1]:


#image読み込み
from sklearn.datasets import load_sample_image


# In[2]:


import tensorflow as tf


# In[3]:


import numpy as np


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[5]:


#サンプル画像のロード
china = load_sample_image("china.jpg") / 255
flower = load_sample_image("flower.jpg") / 255
images = np.array([china, flower])
batch_size, height, width, channels = images.shape


# In[6]:


images.shape


# In[7]:


china.shape


# In[8]:


flower.shape


# In[9]:


#2個のフィルターの作成
filters = np.zeros(shape = (7,7,channels,2), dtype=np.float32) #7x7のフィルターの空箱
filters[:,3,:,0] = 1 #縦線 [3列(中央)が白でそれ以外がない]
filters[3:,:,:,1] = 1 #横線[3行(中央)が白でそれ以外がない]

outputs = tf.nn.conv2d(images,filters,strides=1,padding="SAME")
#画像に、フィルタをかけて、ストライド1でZeropaddingで畳み込み
#strides は整数か4要素一次元配列[1,sh,sw,1] (1は決め打ち)


# In[10]:


plt.imshow(outputs[0,:,:,1],cmap="gray") #第1画像の第2特徴量マップをプロット


# In[11]:


plt.imshow(china)


# In[12]:


plt.imshow(outputs[1,:,:,1],cmap="gray") #第2画像の第2特徴量マップをプロット


# In[13]:


plt.imshow(flower)


# In[14]:


import keras


# In[15]:


#実際の学習では通常訓練可能変数としてフィルタを定義し、ニューラルネットが最もうまく機能するフィルタを学習できるようにする
conv = keras.layers.Conv2D(filters=32,kernel_size=3,strides=1,padding="same",activation="relu")


# In[16]:


##メモリ爆発


# In[17]:


#pooling層
#平行移動の変換不変


# In[18]:


#max_pooling
max_pool = keras.layers.MaxPool2D(pool_size=2) #デフォルトはストライド=カーネルサイズ、padding=”valid”


# In[19]:


#深度方向のpooling
#output = tf.nn.max_pool(images, ksize=(1,1,1,3), strides=(1,1,1,3), padding="VALID") #3が深度方向のカーネルサイズ
#特徴量マップの並び方によるんじゃないの？？


# In[20]:


#グローバル平均プーリング層
global_avg_pool = keras.layers.GlobalAvgPool2D()


# In[21]:


#CNNのアーキテクチャ(p.456)
model = keras.models.Sequential()

model.add(keras.layers.Conv2D(64, 7, activation="relu", padding="same", input_shape=[28,28,1]))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Conv2D(128, 3, activation="relu", padding="same"))
model.add(keras.layers.Conv2D(128, 3, activation="relu", padding="same"))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Conv2D(256, 3, activation="relu", padding="same"))
model.add(keras.layers.Conv2D(256, 3, activation="relu", padding="same"))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(64,activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10, activation="softmax"))


# In[22]:


#dataset
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()


# In[23]:


plt.plot(y_train_full)


# In[24]:


#scaling, validation_set の準備
# X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
# y_valid, y_train = y_train_full[:5000] , y_train_full[5000:] 
#60000のうちindex0~4999をvalidationsetに分ける→学習が1epoch20分くらいかかったのでデータを大幅に削減

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000] , y_train_full[5000:] 


# In[25]:


class_names = ["T-shirt/top", "Trouser", 'Pullover', "Dress", "Coat",
                        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


# In[26]:


class_names[y_train[0]]


# In[27]:


model.summary()


# In[28]:


model.layers


# In[29]:


model.layers[1].name


# In[30]:


weights, biases = model.layers[0].get_weights()


# In[31]:


weights


# In[32]:


weights.shape


# In[33]:


biases


# In[34]:


biases.shape


# In[35]:


#初期値が入っている


# In[36]:


model.compile(loss="sparse_categorical_crossentropy",
                      optimizer="sgd",
#              optimizer=keras.optimizers.SGD(lr=???), #error
              metrics = ["accuracy"])
#ラベルが疎、クラスが相互排他的
#クラスごとのターゲット確率を計算する場合はcategorical_crossentropyを使う。


# In[38]:


history = model.fit(X_train.reshape(55000,28,28,1),y_train, epochs=30, validation_data=(X_valid.reshape(5000,28,28,1), y_valid))


# In[39]:


# CODE HERE
from tensorflow.keras.models import load_model


# In[40]:


model.save('my_model_fashionmnist_all.h5')  # creates a HDF5 file 'my_model.h5'
#later_model = load_model('my_model_fashionmnist.h5')


# In[42]:


#history = later_model.fit(X_train.reshape(9000,28,28,1),y_train, epochs=10, validation_data=(X_valid.reshape(1000,28,28,1), y_valid))


# In[41]:


import pandas as pd


# In[42]:


#学習履歴をplot
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1) #Y軸範囲を0~1 #gca()はGet Current Axes最後に操作した（Current）Axes
#訓練セットの値は各epoch中の移動平均、valセットの値は各epoch終了時に計算したもの


# In[43]:


model.evaluate(X_test.reshape(10000,28,28,1), y_test)


# In[45]:


#予測例
X_new = X_test[:3]
y_proba = model.predict(X_new.reshape(3,28,28,1))
y_proba


# In[46]:


y_pred = model.predict_classes(X_new.reshape(3,28,28,1))


# In[47]:


np.array(class_names)[y_pred]


# In[48]:


#適用


# In[49]:


import glob

files = glob.glob("./sample/*")
for file in files:
    print(file)


# In[50]:


tes = np.array('sample/IMG_20201208_220156.jpg')


# In[51]:


from PIL import Image


# In[53]:


img = Image.open('sample/IMG_20201208_220156.jpg')
img = img.convert('L')
img = img.resize((28,28))


# In[58]:


img_n = np.array(img)
plt.imshow(img_n,cmap='gray')


# In[56]:


y_p = model.predict(img_n.reshape(1,28,28,1))
y_p


# In[57]:


class_names[8]


# In[ ]:




