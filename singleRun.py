from steamUserReview import *
import json
import os
import numpy as np
import itertools
import time
import seaborn as sns

#Load data
x_train, y_train, x_test, y_test = importData("../train.csv")

# Tokenize
if os.path.exists("tokenizer.json"):
    with open('tokenizer.json') as f:
        tokenizer_json = json.load(f)
else:
    tokenizer_json = ""

x_train, y_train, max_length, tokenizer_json  = tokenize(x_train, y_train, tokenizer_json)
x_test, y_train, _, _ = tokenize(x_test, y_train, tokenizer_json, max_length)

if not os.path.exists("tokenizer.json"):
    with open('tokenizer.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))

optimizer = "RMSprop"
learningRate = 0.001
hiddenLayer = 3
nodeCount = 5
epochs = 6
activation = "relu"
bachSize = 64


model = createModel(max_length, nodeCount, hiddenLayer, activation, optimizer, learningRate)
model.summary()
model, history = fitModel(model, x_train, y_train, epochs, bachSize)

accuracyScore, confusionMatrix = runTest(model, x_test, y_test)

print("Accuracy score is {}% ".format(accuracyScore))

plt.plot(history.history["loss"])
plt.ylabel("loss")
plt.show()

# Plot confusion matrix
plt.subplots()
sns.heatmap(confusionMatrix, annot=True, linewidths=1.5, fmt=".1f")
plt.xlabel("Prediction")
plt.ylabel("Actual")
plt.show()