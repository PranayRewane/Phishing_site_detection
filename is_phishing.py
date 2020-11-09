import detect_features
import phising
def pred_site(url):
    cont="c"
    while(cont!="s"):
        print("enter a url")
        url=input()
        res=detect_features.generate_data_set(url)
        res = np.array(res).reshape(1,-1)
        pred=phising.classifier.predict(res)
        isphishing=pred[0]
        print(isphishing)
        if isphishing==-1:  
            print("phishing site")
        else:
            print("not a phishing site")
        print("press s to stop and c to continue")
        cont=input()
