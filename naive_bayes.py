  #	Naïve Bayesian Classifer- 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
import detect_features
import numpy as np
from sklearn.naive_bayes import GaussianNB
import time
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict 


def load_data():
    

    #Get the data
    training_data = np.genfromtxt(r'dataset.csv', delimiter=',', dtype=np.int32)

    inputs = training_data[:,:-1]         #  inputs - All rows and all columns except the last one  cause its the answer

    outputs = training_data[:,-1]         # Get the labels

    # Divide the data set into training and testing. Total=2456
    
    training_inputs = inputs[:7000]       #  first 7000 rows  excluding last column
    training_outputs = outputs[:7000]     #   first 7000 rows  with only last column
    testing_inputs = inputs[7000:]		  #  remaining rows excluding last column
    testing_outputs = outputs[7000:]      # remaining rows  with only last column

    # Return the four arrays
    return training_inputs, training_outputs, testing_inputs, testing_outputs
   
 
if __name__ == '__main__':        # Entry point of the program
    start_time = time.time()
    train_inputs, train_outputs, test_inputs, test_outputs = load_data()      # get  the data 
	
    classifier= GaussianNB()
  
    classifier.fit(train_inputs, train_outputs)       # Train the classifier model
    
    predictions = classifier.predict(test_inputs)      # make the predictions on testing data
    kfold=KFold(n_splits=10)
    y_pred=cross_val_predict(classifier,train_inputs,train_outputs,cv=kfold)
    cv_res=cross_val_score(classifier,train_inputs,train_outputs,cv=kfold,scoring="accuracy")
    print("K Cross validation accuracy is-> ")
    print(cv_res.mean())
    confusionmatrix=confusion_matrix(test_outputs,predictions)       # Create a confusion matrix 
   
    accuracy = 100.0 * accuracy_score(test_outputs, predictions)     # Calculate the accuracy
    print ("The accuracy of Naive Bayesian classifier " + str(round(accuracy,2))+ "%")
    print("confusion matrix=\n",confusionmatrix)
	
    error=(1-accuracy/100.0)*100.0
    print("The error rate of the Naive Bayesian classifier  " + str(round(error,2)) + "%")
	
  #  report=classification_report(test_outputs,predictions)
  #  print("The classification report is:\n "+ str(report))
    print("Time = %s seconds " % (time.time() - start_time))
    cont="c"
    while(cont!="s"):
    	print("enter a url")
    	url=input()
    	res=detect_features.generate_data_set(url)
    	res = np.array(res).reshape(1,-1)
    	pred=classifier.predict(res)
    	isphishing=pred[0]
    	print(isphishing)
    	if isphishing==1:	
    		print("not a phishing site")
    	else:
        	print("phishing site")
    	print("press s to stop and c to continue")
    	cont=input()


    
    
