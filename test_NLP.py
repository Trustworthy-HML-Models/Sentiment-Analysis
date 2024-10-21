# import required packages
from glob import glob
import os,re,string
import numpy as np
from tensorflow import keras
import pickle

# Loading test dataset from the folder
def load_dataset(file_path, folders):
    texts,labels = [],[]
    for i,label in enumerate(folders):
        for fname in glob(os.path.join(file_path, label, '*.*')):
            texts.append(open(fname, 'r',encoding="utf8").read())
            labels.append(i)
    
    return texts, np.array(labels).astype(np.int64)

#Pre-processing steps
REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
NO_SPACE = ""
SPACE = " "

def preprocess_reviews(reviews):
    
    reviews = [REPLACE_NO_SPACE.sub(NO_SPACE, line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(SPACE, line) for line in reviews]
    
    return reviews


# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow


if __name__ == "__main__": 
	# 1. Load your saved model
	file_name = "models/Group46_NLP_model.h5"
	svm = pickle.load(open(file_name,'rb'))
	final = pickle.load(open(file_name,'rb'))
	# 2. Load your testing data
	#Specyfing the file path as per the problem statement
	file_path ='./data/aclImdb/'
	#specifying both the postive and negative files required for the analysis
	files = ['neg','pos']
	x_train,y_train = load_dataset(f'{file_path}train',files)
	reviews_train_clean = preprocess_reviews(x_train)
	x_test,y_test = load_dataset(f'{file_path}test',files)
	reviews_test_clean = preprocess_reviews(x_test)

	# 3. Run prediction on the test data and print the test accuracy
	import sklearn
	from sklearn.feature_extraction.text import CountVectorizer
	from sklearn.metrics import accuracy_score


	#By removing stop words, we remove the low-level information from our text in order to give more focus to the important information.
	stop_words = ['in', 'of', 'at', 'a', 'the']

	ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 3), stop_words=stop_words)
	ngram_vectorizer.fit(reviews_train_clean)
	X = ngram_vectorizer.transform(reviews_train_clean)
	X_test = ngram_vectorizer.transform(reviews_test_clean)

	final.fit(X,y_test)
	print ("Accuracy of test set: %s" 
       % accuracy_score(y_test, final.predict(X_test)))
