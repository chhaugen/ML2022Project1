from steamUserReview import *
import json
import os
import numpy as np
import itertools
import time

optimizerAlgorithmOptions = ["adam", "SGD", "RMSprop"]
learningRateOptions = [0.1, 0.01, 0.001, 0.0001]
hiddenLayerOptions = [1, 2, 3]
nodeCountOptions = [5, 10, 15]
epochsOptions = [6, 10, 15]
activationOptions = ["relu", "selu"]
bachSizeOptions = [32, 64, 128]

optionNames = [
    "optimizer",
    "learningRate",
    "hiddenLayer",
    "nodeCount",
    "epochs",
    "activation",
    "bachSize",
]

allOptions = [{optionNames[idx]:option for idx, option in enumerate(optionSet)} for optionSet in itertools.product(
                optimizerAlgorithmOptions,
                learningRateOptions,
                hiddenLayerOptions,
                nodeCountOptions,
                epochsOptions,
                activationOptions,
                bachSizeOptions,
                 )]

def main():

    x_train, y_train, x_test, y_test = importData("train.csv")

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
    
    allOptionsCount = len(allOptions)
    remainingOptions = []
    for index, opts in enumerate(allOptions):
        dirPath = f"results/optimizer_{opts['optimizer']}/learningRate_{opts['learningRate']}/hiddenLayer_{opts['hiddenLayer']}/nodeCount_{opts['nodeCount']}/epochs_{opts['epochs']}/activation_{opts['activation']}/bachSize_{opts['bachSize']}"
        dirPath = os.path.abspath(dirPath)
        if not os.path.exists(dirPath):
            remainingOptions.append(opts)
            continue
        try:
            if len(os.listdir(dirPath)) == 0:
                remainingOptions.append(opts)
                continue
        except FileNotFoundError:
            remainingOptions.append(opts)
            continue
    remainingOptionsCount = len(remainingOptions)
    print(f"Skiped {allOptionsCount - remainingOptionsCount} allready done tasks. {remainingOptionsCount} tasks remaining.")

    timeList = []
    
    for index, opts in enumerate(remainingOptions):
        print(f"Started optimizer:{opts['optimizer']}, learningRate:{opts['learningRate']}, hiddenLayer:{opts['hiddenLayer']}, nodeCount:{opts['nodeCount']}, epochs:{opts['epochs']}, activation:{opts['activation']}, bachSize:{opts['bachSize']}")
        dirPath = f"results/optimizer_{opts['optimizer']}/learningRate_{opts['learningRate']}/hiddenLayer_{opts['hiddenLayer']}/nodeCount_{opts['nodeCount']}/epochs_{opts['epochs']}/activation_{opts['activation']}/bachSize_{opts['bachSize']}"
        dirPath = os.path.abspath(dirPath)
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)

        # To not repeat tasks:
        if len(os.listdir(dirPath)) == 0:
            if len(timeList) == 0:
                timeList.append(time.time())
            
            model = createModel(max_length, opts['nodeCount'], opts['hiddenLayer'], opts['activation'], opts['optimizer'], opts['learningRate'])
            if __debug__:
                model.summary()
            model, history = fitModel(model, x_train, y_train, opts['epochs'], opts['bachSize'])
            model.save(f'{dirPath}/model')

            accuracy_score, confusion_matrix = runTest(model, x_test, y_test)

            with open(f'{dirPath}/testData.json', 'w') as f:
                data = {"accuracy_score": accuracy_score, "confusion_matrix": confusion_matrix.tolist()}
                json.dump(data, f)

            with open(f'{dirPath}/history.json', 'w') as f:
                json.dump(history.history, f)
            
            print(f"Saved optimizer:{opts['optimizer']}, learningRate:{opts['learningRate']}, hiddenLayer:{opts['hiddenLayer']}, nodeCount:{opts['nodeCount']}, epochs:{opts['epochs']}, activation:{opts['activation']}, bachSize:{opts['bachSize']}")
            timeList.append(time.time())
            avgMinutes = averageSecounds(timeList) / 60
            print(f"Done with {int(((index+1)*100)/remainingOptionsCount)}% of the work ({index+1}/{remainingOptionsCount}). Average minutes per task is {avgMinutes}. Estimated time remaining is {avgMinutes*(remainingOptionsCount-(index+1))} ")
        else:
            print(f"Skiped optimizer:{opts['optimizer']}, learningRate:{opts['learningRate']}, hiddenLayer:{opts['hiddenLayer']}, nodeCount:{opts['nodeCount']}, epochs:{opts['epochs']}, activation:{opts['activation']}, bachSize:{opts['bachSize']}")
        

def averageSecounds(datetimes):
    timediffList = []
    for index, n in enumerate(datetimes):
        if index != 0:
            timediffList.append(n - prev_time)
        prev_time = n
    return np.mean(timediffList)



#    for activator in activatorOptions:
#        for nodeCount in nodeCountOptions:
#            for epochs in epochsOptions:
#                for bachSize in bachSizeOptions:
#                    for hiddenLayer in hiddenLayerOptions:
#                        print(f"Started activator:{activator}, nodeCount:{nodeCount}, epochs:{epochs}, bachSize:{bachSize}, hiddenLayer:{hiddenLayer}")
#                        dirPath = f"activator_{activator}/nodeCount_{nodeCount}/epochs_{epochs}/bachSize_{bachSize}/hiddenLayer_{hiddenLayer}"
#                        dirPath = os.path.abspath(dirPath)
#                        if not os.path.exists(dirPath):
#                            os.makedirs(dirPath)
#
#                        # To not repeat tasks:
#                        if len(os.listdir(dirPath)) == 0:
#                            model = createModel(max_length, nodeCount, hiddenLayer, activator)
#                            model, history = fitModel(model, x_train, y_train, epochs, bachSize)
#                            model.save(f'{dirPath}/model')
#
#                            with open(f'{dirPath}/history.json', 'w') as f:
#                                json.dump(history.history, f)
#                            
#                            print(f"Saved activator:{activator}, nodeCount:{nodeCount}, epochs:{epochs}, bachSize:{bachSize}, hiddenLayer:{hiddenLayer}")
#                        else:
#                            print(f"Skiped activator:{activator}, nodeCount:{nodeCount}, epochs:{epochs}, bachSize:{bachSize}, hiddenLayer:{hiddenLayer}")

if __name__ == "__main__":
    main()