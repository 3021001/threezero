import sys
!{sys.executable} -m pip install keras pandas numpy image matplotlib scikit-learn

//warning출력 off
import warnings
warnings.filterwarnings('ignore')

//필요한 패키지 import
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random 
import os
print(os.listdir("./data"))

//글로벌 변수 선언
#Define constants
FAST_RUN = False
IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3

//파일명과 정답 설정
#Preparing Traning Data
filenames = os.listdir("./data/train")
categories = []
for filename in filenames :
  category = filename.split('.')[0]
  if category == 'dog' :
    categories.append(1)
  else :
    categories.append(0)
    
df = pd.DataFrame({
  'filename':filenames,
  'category':categories
})

//확인
df.head

//데이터 balance 확인
#See total in count
df['category'].value_counts().plot.bar()

//Sample 데이터 확인
#See sample Image
sample = random.choice(filenames)
Image = load_img("./data/train/"+sample)
plt.imshow(image)

//신경망 모델 구성하기
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax')) 

model.complie(loss='categorical_crossentropy', optimizer='rmsprop', mertrics=['accuracy'])

model.summary()

//콜백 정의
#Callbacks
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

//Early Stopping 정의
#Early Stop
#To prevent over fitting we will stop the learning after 10 epochs and val_loss value not decreased
earlystop = EarlyStopping(patience=10)

//Learning Rate조정 정의
#Learning Rate Reductuon
#We will reduce the learning rate when then accuracy not increase for 2 steps
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=2,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

//callback 설정
callbacks = [earlystop, learning_rate_reduction]

//개,고양이를 string으로 변환
df["category"] = df["category"].replace({0:'cat',1:'dog'})

//train데이터의 분포 확인
train_df['category'].value.counts().plot.bar()

//validation 분포 확인
validation_df['category'].value_counts().plot.bar()

//학습,검증데이터 확인
total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size=15

//학습데이터 augmetation
#Train Generator
train_datagen = ImageDataGenerator(
  rotation_range=15,
  rescale=1./255,
  shear_rang=0.1,
  zoom_range=0.2,
  horizontal_flip=True,
  width_shift_range=0.1,
  height_shift_range=0.1
)
train_generator = train_datagen.flow_from_dataframe(
  train_df,
  "./data/train/",
  x_col='filename',
  y_col='category',
  taget_size=IMAGE_ZISE,
  class_mode='categorical',
  batch_size=batch_size
)

//Validation augmentation 작업
validataion_datagen = ImageDataGenerator(rescale=1./255)
validataion_generator = validation_datagen.flow_from_dataframe(
  validata_df,
  "./data/train/",
  x_col='filename',
  y_col='category',
  target_size=IMAGE_SIZE,
  class_mode='categorical',
  batch_size=batch_size
)

//샘플 확인
#See how our generator work
example_df = train_df.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(
  example_df,
  "./data/train/",
  x_col='filename',
  y_col='category',
  target_size=IMAGE_SIZE,
  class_mode='categorical'
)

//이미지 확인
plt.figure(figsize=(12,12))
for i in range(0,15):
  plt.subplot(5, 3, i+1)
  for X_batch, Y_batch in example_gernerator:
    image = X_batch[0]
    plt.imshow(image)
    break
plt.tight_layout()
plt.show



