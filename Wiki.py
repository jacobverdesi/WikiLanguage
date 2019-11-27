import numpy as np
import pandas as pd
import Classifyers
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

def classify(file):
    classifyers = [Classifyers.consinants,Classifyers.containsDE]
    data=[[]]
    for line in file:
        lines = line.split("|")
        if len(lines)==2:
            classifiedLine=[]
            classifiedLine.append(lines[0])
            for func in classifyers:
                classifiedLine.append(func(lines[1]))
        else:
            classifiedLine=[]
            for func in classifyers:
                classifiedLine.append(func(lines[0]))
        data.append(classifiedLine)
    data=data[1:]
    return np.array(data)
def train(examples,hypothesisOut):
    trained=classify(open(examples).readlines())
    attr = np.arange(1, 10)
    np.savetxt("train.cld",trained,fmt='%s')
    dtrain = pd.read_csv("train.cld", sep=" ", names=attr)
    trainFeatures=[]
    for i in range(2,np.shape(trained)[1]+1):
        trainFeatures.append(i)
    x_train = dtrain[trainFeatures]
    y_train = dtrain[1]
    clf = DecisionTreeClassifier(criterion="entropy")
    clf.fit(x_train, y_train)
    with open(hypothesisOut,'wb') as handle:
        pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)
def predict(hypothesis,testFile):
    attr = np.arange(1, 10)
    with open(hypothesis,'rb') as handle:
        clf = pickle.load(handle)
    test=classify(open(testFile).readlines())
    np.savetxt("test.cld",test,fmt='%s')
    dtest=pd.read_csv("test.cld", sep=" ", names=attr)
    testFeatures = []
    for i in range(1, np.shape(test)[1]+1):
        testFeatures.append(i)
    x_test=dtest[testFeatures]
    y_pred3 = clf.predict(x_test)
    return y_pred3

def accuracy(answer,y_pred):
    y_test = np.loadtxt(answer, dtype=np.object)
    print("Prediction: ",end='')
    print(y_pred)
    print("Expected: ",end='')
    print(y_test)
    print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))

def main():
    trainedFile="Data/train.dat"
    testAnswer="Data/answer.out"
    train(trainedFile,"clf.txt")
    pred=predict("clf.txt","Data/test.dat")
    accuracy(testAnswer,pred)
if __name__ == '__main__':
    main()