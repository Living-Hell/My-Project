# IoT Intrusion Detection [Keylogging] CNN-img
## Importing Libraries
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
#import libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD,Adam
import keras
# Conv1D + LSTM
from keras.layers.convolutional import Conv1D,MaxPooling1D,Conv2D,MaxPooling2D
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten

from tensorflow.keras import models, layers

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import EfficientNetB6
from tensorflow.keras.applications import InceptionV3

from tensorflow.keras.applications import ResNet50
```

## Reading Data
```
df_dk=pd.read_csv('../input/keylogger-detection/Keylogger_Detection.csv')
df_dk.head(5)
```

## Data Preparation
```
df_dk.dtypes
Class_df = df_dk["Class"]
df_num = df_dk.select_dtypes(include=[np.number])
df_num = df_num.join(Class_df)
```
### Removing Columns
```
df_num.drop('Unnamed: 0', axis=1, inplace=True)
```
### Removing nan values
```
print(df_num.isna().sum().sum())
df_num=df_num.dropna()
```
```
df_num["label"]=df_num["Class"]
df_num.loc[df_num.label == "Benign", 'label'] = 0
df_num.loc[df_num.label == "Keylogger", 'label'] = 1
df_num.drop('Class', axis=1, inplace=True)
df_num.sample(5)
```

## Distribution Classes
```
df_num=df_num.sample(n=200000)
df_num.groupby('label').size()
```

## Remove Useless Features
```
for col in (df_num.iloc[:,:-1].columns):
    if(df_num[col].min()==df_num[col].max()):
            df_num.drop(col, axis=1, inplace=True)
```

## Feature Scaling
```
# Normalization OR Standardization
def standardize(df,col):
    #df[col]= (df[col]-df[col].mean())/(df[col].std()) # Standardization
    df[col]= 255*(df[col]-df[col].min())/(df[col].max()-df[col].min()) #Normalization

for i in (df_num.iloc[:,:-1].columns):
    standardize (df_num,i)

df_num.head()

for i in range(957):
    col="A"+str(i)
    df_num[col]=0

df_num.shape

target=['label']
features = [c for c in df_num.columns if c!="label"]

row_1=df_num[features].iloc[0].to_numpy()
row_1.shape
row_2=row_1.reshape(32, 32)
row_2=row_2.astype(int)
print(type(row_2))
from matplotlib import pyplot as plt
plt.imshow(row_2,cmap='gray')
plt.show()
```
## Split DataSet
```
X = df_num[features].values # Features
y = df_num[target].values # Target

X=X.astype(np.float32)
y=y.astype(np.float32)

X.shape,y.shape,

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

X_train.shape,X_test.shape,y_train.shape,y_test.shape

X_train_cnn = np.reshape(X_train, (X_train.shape[0], 32,32,1))
X_test_cnn = np.reshape(X_test, (X_test.shape[0], 32,32,1))
print(X_train_cnn.shape)
print(X_test_cnn.shape)
```

## Build CNN Model
```
learning_rate=0.0001
batch_size=1024
epochs = 50

model_save = ModelCheckpoint('./Keylogging.h5', 
                             save_best_only = True, 
                             save_weights_only = True,
                             monitor = 'val_loss', 
                             mode = 'min', verbose = 1)
early_stop = EarlyStopping(monitor = 'val_loss', min_delta = 0.0001, 
                           patience = 10, mode = 'min', verbose = 1,
                           restore_best_weights = True)
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.75, 
                              patience = 10, min_delta = 0.0001, 
                              mode = 'min', verbose = 1)

def create_model(): 
    inputs = layers.Input(shape=(32,32,1))
    efficientnet_layers = EfficientNetB0(include_top=False,input_shape=(),weights='imagenet',pooling='avg')
    model = Sequential()
    
    model.add(inputs)
    model.add(keras.layers.Conv2D(3,3,activation='relu',padding='same'))
    model.add(efficientnet_layers)
    #model.add(Dropout(0.3))
    model.add(Dense(1, activation="sigmoid"))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=[keras.metrics.AUC(name='auc')])

    return model

model = create_model()
model.summary()

history = model.fit(X_train_cnn,
                    y_train,
                    batch_size=batch_size,
                    steps_per_epoch=X_train.shape[0] // batch_size,
                    epochs=epochs,
                    validation_data=(X_test_cnn,y_test),
                    callbacks = [model_save, early_stop, reduce_lr],)
```

## Evaluation
```
y_pred = model.predict(X_test_cnn, batch_size=512)
AUC = metrics.roc_auc_score(y_test,y_pred)
print("AUC: {:.3f}".format(AUC))

hist_df = pd.DataFrame(history.history)
hist_df.to_csv('history.csv')
```

### Training Curves
```
plt.figure(figsize=(15,5))
plt.plot(range(history.epoch[-1]+1),history.history['val_auc'],label='val_auc')
plt.plot(range(history.epoch[-1]+1),history.history['auc'],label='auc')
plt.title('auc'); plt.xlabel('Epoch'); plt.ylabel('auc');plt.legend(); 
plt.show()

plt.figure(figsize=(15,5))
plt.plot(range(history.epoch[-1]+1),history.history['val_loss'],label='Val_loss')
plt.plot(range(history.epoch[-1]+1),history.history['loss'],label='loss')
plt.title('loss'); plt.xlabel('Epoch'); plt.ylabel('loss');plt.legend(); 
plt.show()
```

### ROC Curve
```
def generate_results(y_test, y_score):
    # print(y_score)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    
generate_results(y_test, y_pred)
```

# Resources:
- [URL1](https://medium.com/analytics-vidhya/cnn-based-malware-detection-python-and-tensorflow-717f8de84ee)
- [URL2](https://medium.com/analytics-vidhya/malware-detection-with-deep-learning-state-of-the-art-177c81aa83ea)
