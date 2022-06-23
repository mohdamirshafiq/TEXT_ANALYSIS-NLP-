# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 09:27:37 2022

@author: End User
"""
import os
import re
import json
import pickle
import datetime
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import LSTM,Dense,Dropout,Bidirectional,Embedding
from tensorflow.keras.preprocessing.text import Tokenizer,tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Input

from modules_for_nlp import model_training,model_evaluation

#%% Static
log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
CSV_URL = "https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv"
LOG_FOLDER_PATH = os.path.join(os.getcwd(),'logs',log_dir)
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'saved_models','model.h5')
TOKENIZER_PATH = os.path.join(os.getcwd(),'saved_models','tokenizer_text.json')
OHE_PATH = os.path.join(os.getcwd(),'saved_models','ohe.pkl')

vocab_size = 10000
oov_token = 'OOV'
max_len = 400

#%% EDA
# Step 1) Data loading
df = pd.read_csv(CSV_URL)
#%% For backup purpose
df_copy = df.copy() # For backup

#%%
# Step 2) Data inspection
df.head(10)
df.tail(10)
df.info()
df.describe()

# To find out the unique targets
df['category'].unique()
    # The categories are:
        # 0. tech
        # 1. business
        # 2. sport
        # 3. entertainment
        # 4. politics

# To test out the text and category
df['text'][5]
df['category'][5]

# Check for duplicates
df.duplicated().sum()
df[df.duplicated()] # To find out more about the duplicates
    # There are 99 duplicates available

# Check for missing values
df.isna().sum()
    # No missing values found

# Step 3) Data cleaning
    # Things to be filtered:
        # 1. Filter numbers since it seems unnecessary
        # 2. Need to remove duplicates

# To remove duplicated data
df = df.drop_duplicates()

# To remove numbers and converting to small letters (just to make sure no capitals)
text = df['text'].values # Features: X
category = df['category'].values # category: y

for index,txt in enumerate(text):
    # convert text into lower case
    # remove numbers
    text[index] = re.sub('[^a-zA-Z]',' ',txt).split()

# Step 4) Features selection
# Nothing to select

#%%
# Step 5) Preprocessing
        # 1) Tokenization
tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_token)

# To learn all of the words
tokenizer.fit_on_texts(text)

word_index = tokenizer.word_index
print(word_index)

# To convert into numbers
train_sequences = tokenizer.texts_to_sequences(text)

        # 2) Padding and truncating (to make sure all of them in equal length numbers)
# List comprehension
length_of_text = [len(i) for i in train_sequences]

# To get the number of max length for padding
print(np.median(length_of_text))

# Padding - This will become X
padded_text = pad_sequences(train_sequences,maxlen=max_len,
              padding='post',
              truncating='post')

        # 3) One Hot Encoding for the Target
ohe = OneHotEncoder(sparse=False)
# This will become y
category = ohe.fit_transform(np.expand_dims(category,axis=-1))

        # 4) Train test split
X_train,X_test,y_train,y_test = train_test_split(padded_text,category,
                                                 test_size=0.3,
                                                 random_state=123)

# To make sure X_train & X_test in 3 dimensional, expand it
X_train = np.expand_dims(X_train,axis=-1)
X_test = np.expand_dims(X_test,axis=-1)

#%% Model development
    # Use LSTM layers,dropout,dense,input
    # ahieve >70% accuracy, and F1 score >0.7
    # Input shape --> np.shape(X_train)[1:] --> (340,1)

# Sequential approach
embedding_dim = 128
model = Sequential()
model.add(Input(shape=(400)))
model.add(Embedding(vocab_size,embedding_dim))
model.add(Bidirectional(LSTM(embedding_dim,return_sequences=(True))))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(5,'softmax'))
model.summary()

# To plot model architecture
plot_model(model)

# To compile model
model.compile(optimizer='adam',loss='categorical_crossentropy',
              metrics='acc')

#%%
# callbacks
tensorboard_callback = TensorBoard(log_dir=LOG_FOLDER_PATH)
early_stopping_callback = EarlyStopping(monitor='loss',patience=5)

# Model training
hist = model.fit(X_train,y_train,epochs=100,
          batch_size=128,
          validation_data=(X_test,y_test),
          callbacks=tensorboard_callback)

#%%
mod_train = model_training()
hist.history.keys()

# Plotting the graph of history keys
mod_train.plot_history_keys(hist)

#%% Model Evaluation
y_true = y_test
y_pred = model.predict(X_test)

#%%
# Since classification metrics can't handle a mix of multilabel-indicator 
# and continuous-multioutput targets, use argmax against y_true & y_pred
y_true = np.argmax(y_true,axis=1)
y_pred = np.argmax(y_pred,axis=1)

# To print classification_report,accuracy_score,confusion_matrix
mod_evaluate = model_evaluation()
mod_evaluate.evaluation_reports(y_true, y_pred)

#%% Saving Model
# To save model
model.save(MODEL_SAVE_PATH)

# To save tokenizer
token_json = tokenizer.to_json()
with open(TOKENIZER_PATH,'w') as json_file:
    json.dump(token_json,json_file)

# To save OneHotEncoder
with open(OHE_PATH,'wb') as file:
    pickle.dump(ohe,file)


#%% Model deployment
# To load trained model
loaded_model = load_model(os.path.join(os.getcwd(),'saved_models','model.h5'))
loaded_model.summary()

# To load tokenizer
with open(TOKENIZER_PATH,'r') as json_file:
    loaded_tokenizer = json.load(json_file)

# To load OneHotEncoder
with open(OHE_PATH,'rb') as file:
    loaded_ohe = pickle.load(file)

#%%
input_text = 'jobs growth still slow in the us the us created fewer jobs than expected in january  but a fall in jobseekers pushed the unemployment rate to its lowest level in three years.  according to labor department figures  us firms added only 146 000 jobs in january. the gain in non-farm payrolls was below market expectations of 190 000 new jobs. nevertheless it was enough to push down the unemployment rate to 5.2%  its lowest level since september 2001. the job gains mean that president bush can celebrate - albeit by a very fine margin - a net growth in jobs in the us economy in his first term in office. he presided over a net fall in jobs up to last november s presidential election - the first president to do so since herbert hoover. as a result  job creation became a key issue in last year s election. however  when adding december and january s figures  the administration s first term jobs record ended in positive territory.  the labor department also said it had revised down the jobs gains in december 2004  from 157 000 to 133 000.  analysts said the growth in new jobs was not as strong as could be expected given the favourable economic conditions.  it suggests that employment is continuing to expand at a moderate pace   said rick egelton  deputy chief economist at bmo financial group.  we are not getting the boost to employment that we would have got given the low value of the dollar and the still relatively low interest rate environment.   the economy is producing a moderate but not a satisfying amount of job growth   said ken mayland  president of clearview economics.  that means there are a limited number of new opportunities for workers.'
# input_text = input('Put the article here')
# Preprocessing the input
input_text = re.sub('[^a-zA-Z]',' ',input_text).split()

tokenizer = tokenizer_from_json(loaded_tokenizer)
input_text_encoded = tokenizer.texts_to_sequences(input_text)
input_text_encoded = pad_sequences(np.array(input_text_encoded).T,maxlen=400,
                                    padding='post',
                                    truncating='post')

# To predict results
outcome = loaded_model.predict(np.expand_dims(input_text_encoded,axis=-1))

print(loaded_ohe.inverse_transform(outcome))

#%% Discussion/ Reporting
# Model achieved around 82% accuracy during training.
# However the model show overfitting when the acc performs better than val_acc.
# with acc at 0.9825 and val_acc at 0.8213.
# Therefore, EarlyStopping has been implemented.
# May increase dropout rate in future to control overfitting.
# Besides, may try with different Deep Learning architecture to improve the 
# model. For example, implementing BERT model, transformer model or GPT3 model.


