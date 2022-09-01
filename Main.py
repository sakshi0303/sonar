
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter import simpledialog
from tkinter import filedialog
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from PNN import *
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

main = tkinter.Tk()
main.title("Underwater Sonar Signals Recognition by Incremental Data Stream Mining with Conflict Analysis") #designing main screen
main.geometry("1300x1200")

global filename
svm_acc = []
nb_acc = []
nn_acc = []
svm_roc = []
nb_roc = []
nn_roc = []

global X_train1, X_test1, y_train1, y_test1
global X_train2, X_test2, y_train2, y_test2
global X_train3, X_test3, y_train3, y_test3
global cols

def upload():
    text.delete('1.0', END)
    global filename
    global X, Y
    filename = filedialog.askopenfilename(initialdir="dataset")
    data = pd.read_csv(filename)
    rows = data.shape[0]  # gives number of row count
    cols = data.shape[1]  # gives number of col count
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");
    text.insert(END,"Number of rows found in dataset : "+str(len(data))+"\n")
    text.insert(END,"Number of columns/features found in dataset : "+str(cols)+"\n")
    cols = cols - 1
    X = data.values[:, 0:cols] 
    labels = data.values[:, cols]
    Y = []
    for i in range(len(labels)):
        if labels[i] == 'R':
            Y.append(0.0)
        else:
            Y.append(1.0)
    Y = np.asarray(Y)
    
    
def generateStream():
    text.delete('1.0', END)
    global X_train1, X_test1, y_train1, y_test1
    global X_train2, X_test2, y_train2, y_test2
    global X_train3, X_test3, y_train3, y_test3

    X_train1, X_test1, y_train1, y_test1 = train_test_split(X, Y, test_size = 0.1, random_state = 42)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, Y, test_size = 0.2, random_state = 42)
    X_train3, X_test3, y_train3, y_test3 = train_test_split(X, Y, test_size = 0.3, random_state = 42)

    text.insert(END,"Stream 1 training size : "+str(len(X_train1))+" & testing size : "+str(len(X_test1))+"\n")
    text.insert(END,"Stream 2 training size : "+str(len(X_train2))+" & testing size : "+str(len(X_test2))+"\n")
    text.insert(END,"Stream 3 training size : "+str(len(X_train3))+" & testing size : "+str(len(X_test3))+"\n")
    
        
            
def svmAlgorithm():
    text.delete('1.0', END)
    svm_acc.clear()
    svm_roc.clear()

    clf = SGDClassifier(learning_rate = 'constant', eta0 = 0.1, shuffle = False)
    clf.partial_fit(X_train1, y_train1,classes=np.unique(Y))  #SVM training on stream1
    y_pred = clf.predict(X_test1)
    accuracy1 = accuracy_score(y_test1,y_pred)*100
    roc1 = roc_auc_score(y_test1,y_pred)
    svm_acc.append(accuracy1)
    svm_roc.append((roc1*100))

    clf.partial_fit(X_train2, y_train2) #incremental training on stream2
    y_pred = clf.predict(X_test2) 
    accuracy2 = accuracy_score(y_test2,y_pred)*100
    roc2 = roc_auc_score(y_test2,y_pred)
    svm_acc.append(accuracy2)
    svm_roc.append((roc2*100))

    clf.partial_fit(X_train3, y_train3) #incremental training on stream3
    y_pred = clf.predict(X_test3)
    accuracy3 = accuracy_score(y_test3,y_pred)*100
    roc3 = roc_auc_score(y_test3,y_pred)
    svm_acc.append(accuracy3)
    svm_roc.append((roc3*100))

    cm = confusion_matrix(y_test3, y_pred)

    text.insert(END,"SVM Stream 1 accuracy : "+str(accuracy1)+"\n")
    text.insert(END,"SVM Stream 1 ROC : "+str(roc1*100)+"\n\n")
    

    text.insert(END,"SVM Stream 2 accuracy : "+str(accuracy2)+"\n")
    text.insert(END,"SVM Stream 2 ROC : "+str(roc2*100)+"\n\n")

    text.insert(END,"SVM Stream 3 accuracy : "+str(accuracy3)+"\n")
    text.insert(END,"SVM Stream 3 ROC : "+str(roc3*100)+"\n\n")
    text.insert(END,"SVM Confusion Matrix : "+str(cm)+"\n\n")


def naivebayesAlgorithm():
    text.delete('1.0', END)
    nb_acc.clear()
    nb_roc.clear()

    clf = MultinomialNB()
    clf.partial_fit(X_train1, y_train1,classes=np.unique(Y))  #Naive Bayesian training on stream1
    y_pred = clf.predict(X_test1)
    accuracy1 = accuracy_score(y_test1,y_pred)*100
    roc1 = roc_auc_score(y_test1,y_pred)
    nb_acc.append(accuracy1)
    nb_roc.append((roc1*100))

    clf.partial_fit(X_train2, y_train2) #incremental training on stream2
    y_pred = clf.predict(X_test2) 
    accuracy2 = accuracy_score(y_test2,y_pred)*100
    roc2 = roc_auc_score(y_test2,y_pred)
    nb_acc.append(accuracy2)
    nb_roc.append((roc2*100))

    clf.partial_fit(X_train3, y_train3) #incremental training on stream3
    y_pred = clf.predict(X_test3)
    accuracy3 = accuracy_score(y_test3,y_pred)*100
    roc3 = roc_auc_score(y_test3,y_pred)
    nb_acc.append(accuracy3)
    nb_roc.append((roc3*100))

    cm = confusion_matrix(y_test3, y_pred)

    text.insert(END,"Naive Bayesian Stream 1 accuracy : "+str(accuracy1)+"\n")
    text.insert(END,"Naive Bayesian Stream 1 ROC : "+str(roc1*100)+"\n\n")
    

    text.insert(END,"Naive Bayesian Stream 2 accuracy : "+str(accuracy2)+"\n")
    text.insert(END,"Naive Bayesian Stream 2 ROC : "+str(roc2*100)+"\n\n")

    text.insert(END,"Naive Bayesian Stream 3 accuracy : "+str(accuracy3)+"\n")
    text.insert(END,"Naive Bayesian Stream 3 ROC : "+str(roc3*100)+"\n\n")
    text.insert(END,"Naive Bayesian Confusion Matrix : "+str(cm)+"\n\n")

  
def neuralnetworkAlgorithm():
    text.delete('1.0', END)
    nn_acc.clear()
    nn_roc.clear()

    neuralnetwork = broadnet_enhmap(maptimes = 10, 
                       enhencetimes = 10,
                       traintimes = 10,
                       map_function = 'tanh',
                       enhence_function = 'sigmoid',
                       batchsize = 'auto', 
                       acc = 1,
                       mapstep = 10,
                       enhencestep = 10,
                       reg = 0.001)

    neuralnetwork.fit(X_train1, y_train1)
    predictlabel = neuralnetwork.predict(X_test1)
    accuracy1 = show_accuracy(predictlabel,y_test1) * 100
    roc1 = roc_auc_score(y_test1,predictlabel)
    nn_acc.append(accuracy1)
    nn_roc.append((roc1*100))

    neuralnetwork.incremental_input(X_train1, X_train2, y_train2)
    predictlabel = neuralnetwork.predict(X_test2)
    accuracy2 = show_accuracy(predictlabel,y_test2) * 100
    nn_acc.append(accuracy2)
    roc2 = roc_auc_score(y_test2,predictlabel)
    nn_roc.append((roc2*100))

    neuralnetwork.incremental_input(np.row_stack((X_train1, X_train2)), X_train3, y_train3)
    predictlabel = neuralnetwork.predict(X_test3)
    accuracy3 = show_accuracy(predictlabel,y_test3) * 100
    nn_acc.append(accuracy3)
    roc3 = roc_auc_score(y_test3,predictlabel)
    nn_roc.append((roc3*100))

    cm = confusion_matrix(y_test3, predictlabel)

    text.insert(END,"Neural Network Stream 1 accuracy : "+str(accuracy1)+"\n")
    text.insert(END,"Neural Network Stream 1 ROC : "+str(roc1*100)+"\n\n")
    

    text.insert(END,"Neural Network Stream 2 accuracy : "+str(accuracy2)+"\n")
    text.insert(END,"Neural Network Stream 2 ROC : "+str(roc2*100)+"\n\n")

    text.insert(END,"Neural Network Stream 3 accuracy : "+str(accuracy3)+"\n")
    text.insert(END,"Neural Network Stream 3 ROC : "+str(roc3*100)+"\n\n")
    text.insert(END,"Neural Network Confusion Matrix : "+str(cm)+"\n\n")


def accuracygraph():
    plt.plot(svm_acc, label="Incrementing SVM Accuracy")
    plt.plot(nb_acc, label="Incrementing Naive Bayesian Accuracy")
    plt.plot(nn_acc, label="Incrementing Neural Network Accuracy")
    plt.legend(loc='lower left')
    plt.title("SVM, Naive Bayesian, Neural Network Accuracy", fontsize=16, fontweight='bold')
    plt.xlabel("Algorithms")
    plt.ylabel("Accuracy")
    plt.show()
    
def rocgraph():
    plt.plot(svm_roc, label="Incrementing SVM ROC")
    plt.plot(nb_roc, label="Incrementing Naive Bayesian ROC")
    plt.plot(nn_roc, label="Incrementing Neural Network ROC")
    plt.legend(loc='lower left')
    plt.title("SVM, Naive Bayesian, Neural Network ROC", fontsize=16, fontweight='bold')
    plt.xlabel("Algorithms")
    plt.ylabel("ROC")
    plt.show()
    

font = ('times', 16, 'bold')
title = Label(main, text='Underwater Sonar Signals Recognition by Incremental Data Stream Mining with Conflict Analysis')
title.config(bg='LightGoldenrod1', fg='medium orchid')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=480,y=100)
text.config(font=font1)


font1 = ('times', 14, 'bold')
uploadButton = Button(main, text="Upload Sonar Dataset", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

readButton = Button(main, text="Convert Dataset into Streams", command=generateStream)
readButton.place(x=50,y=150)
readButton.config(font=font1) 

svmButton = Button(main, text="Run Increment SVM Algorithm", command=svmAlgorithm)
svmButton.place(x=50,y=200)
svmButton.config(font=font1) 

naivebayesButton = Button(main, text="Run Increment Naive Bayesian Algorithm", command=naivebayesAlgorithm)
naivebayesButton.place(x=50,y=250)
naivebayesButton.config(font=font1) 

neuralnetworkButton = Button(main, text="Run Neural Network Algorithm", command=neuralnetworkAlgorithm)
neuralnetworkButton.place(x=50,y=300)
neuralnetworkButton.config(font=font1)

graphButton = Button(main, text="Accuracy Graph", command=accuracygraph)
graphButton.place(x=50,y=350)
graphButton.config(font=font1)

rocButton = Button(main, text="ROC Graph", command=rocgraph)
rocButton.place(x=50,y=400)
rocButton.config(font=font1) 

main.config(bg='OliveDrab2')
main.mainloop()
