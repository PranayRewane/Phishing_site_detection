from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

import numpy as np
import detect_features
#from sklearn import tree
#from sklearn.tree import export_graphviz
##import pydotplus
#import graphviz

def load_data():
    

    #Get the data
    training_data = np.genfromtxt(r'dataset.csv', delimiter=',', dtype=np.int32)
    
    inputs = training_data[:,:-1]         # Get the inputs - All rows and all columns except the last one 

    
    outputs = training_data[:,-1]         # Get the labels

    # Divide the data set into training and testing. Total=2456
    #  Training dataset (1500 rows)
    #  Training dataset (956 rows) 	
    training_inputs = inputs[:7000]       #  Select first 1500 rows (0-1499) excluding last column
    training_outputs = outputs[:7000]     #  Select first 1500 rows (0-1499) with only last column
    testing_inputs = inputs[7000:]		  #  Select remaining rows (1500-2455) excluding last column
    testing_outputs = outputs[7000:]      #  Select remaining rows (1500-2455) with only last column

    # Return the four arrays
    return training_inputs, training_outputs, testing_inputs, testing_outputs
   

if __name__ == '__main__':        # Entry point of the program
    train_inputs, train_outputs, test_inputs, test_outputs = load_data()      # get  the data 

    classifier = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    classifier.fit(train_inputs, train_outputs)
  
         # Train the classifier model
    
    predictions = classifier.predict(test_inputs)      # make the predictions on testing data
    
    kfold=KFold(n_splits=10)
    y_pred=cross_val_predict(classifier,train_inputs,train_outputs,cv=kfold)
    cv_res=cross_val_score(classifier,train_inputs,train_outputs,cv=kfold,scoring="accuracy")
    print("K Cross validation accuracy is-> ")
    print(cv_res.mean())
    
    
    confusionmatrix=confusion_matrix(test_outputs,predictions)       # Create a confusion matrix 
   
    accuracy = 100.0 * accuracy_score(test_outputs, predictions)     # Calculate the accuracy
    print ("The accuracy of your k-nn on testing data is: " + str(round(accuracy,2))+ "%")
    print("Confusionmatrix=\n",confusionmatrix)
	
    error=(1-accuracy/100.0)*100.0
    print("The error rate of the k-nn on testing data is: " + str(round(error,2)) + "%")
	
    report=classification_report(test_outputs,predictions)
    print("The Classification report is:\n "+ str(report))
    
    # taking user input url and predicting it
    cont="c"
    while(cont!="s"):
        print("Enter a url")
        url=input()
        res=detect_features.generate_data_set(url)
        res = np.array(res).reshape(1,-1)
        pred=classifier.predict(res)
        isphishing=pred[0]
        print(isphishing)
        if isphishing==1:  
            print("Not a phishing site")
        else:
            print("Phishing site")
        print("Press s to stop and c to continue")
        cont=input()

    
