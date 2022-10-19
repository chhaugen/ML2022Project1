from statistics import median
import string
from pandas import Series, read_csv
from tensorflow import keras
from keras import Sequential
from keras.callbacks import History 
from keras.preprocessing.text import Tokenizer, tokenizer_from_json
from keras.layers import Embedding, LSTM, GRU, Dense, Flatten
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error
import numpy as np
import seaborn as sns

VOCABULARY_SIZE = 5000

def importData(path: string):
    data = read_csv("train.csv", header=0)
    train = data.iloc[ : int(len(data) * .90)]
    test = data.iloc[ int(len(data) * .90) : ]
    x_train, y_train = train['user_review'], train['user_suggestion']
    x_test, y_test = test['user_review'], test['user_suggestion']
    return x_train, y_train, x_test, y_test



def tokenize(x, y, tokenizer_json=None, max_length=None):

    if tokenizer_json is None:
        tokenizer = Tokenizer(num_words=VOCABULARY_SIZE)
        tokenizer.fit_on_texts(x)
        tokenizer_json = tokenizer.to_json()
    else:
        tokenizer = tokenizer_from_json(tokenizer_json)
    
    x_tokens = tokenizer.texts_to_sequences(x)

    if max_length is None:
        length_array = [len(i) for i in x_tokens]
        #print(length_array)
        max_length = int(np.percentile(length_array,98))

    x_tokens_padded = pad_sequences(x_tokens, maxlen=max_length)
    x_padded_array = np.array(x_tokens_padded)
    y_array = np.array(y)

    return x_padded_array, y_array, max_length, tokenizer_json

def createModel(max_length, nodeCount, hiddenLayerCount, activation, optimizerName, learningRate):
    model = Sequential()
    # Embedding: Turns positive integers (indexes) into dense vectors of fixed size.
    # e.g. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
    model.add(Embedding(VOCABULARY_SIZE, 16, input_length=max_length))
    #model.add(GRU(150))
    for n in range(hiddenLayerCount):
        # Just your regular densely-connected NN layer.
        model.add(Dense(nodeCount, activation=activation))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # 'binary_crossentropy' Computes the cross-entropy loss between true labels and predicted labels.
    model.compile(loss='binary_crossentropy', optimizer = optimizerByName(optimizerName, learningRate), metrics = ['accuracy'])
    
    return model

def optimizerByName(optimizerName, learningRate):
    if optimizerName == "adam":
        return keras.optimizers.Adam(learning_rate=learningRate)
    elif optimizerName == "SGD":
        return keras.optimizers.SGD(learning_rate=learningRate)
    elif optimizerName == "RMSprop":
        return keras.optimizers.RMSprop(learning_rate=learningRate)

def fitModel(model, x, y, epochs=10, batch_size=64):
    history = History()
    #print(f'count: {len(y)}, 0.1 of data length={len(y)*0.1}')
    model.fit(x, y, validation_split=0.1, epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=[history])
    return model, history

def runTest(model, x_test, y_test):
    y_predicted = ((model.predict(x_test) > 0.5).astype("int32"))
    accuracyScore = accuracy_score(y_pred=y_predicted,y_true=y_test)
    confusionMatrix = confusion_matrix(y_pred=y_predicted,y_true=y_test)
    return accuracyScore, confusionMatrix


