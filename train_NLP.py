# import required packages
from glob import glob
import os,re,string
import numpy as np
from tensorflow import keras
import pickle


# Loading dataset from the folder
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
	# 1. load your training data
	#Specyfing the file path as per the problem statement
	file_path ='./data/aclImdb/'
	#specifying both the postive and negative files required for the analysis
	files = ['neg','pos']
	x_train,y_train = load_dataset(f'{file_path}train',files)
	reviews_train_clean = preprocess_reviews(x_train)
	

	# 2. Train your network
	# 		Make sure to print your training loss and accuracy within training to show progress
	# 		Make sure you print the final training accuracy
	import sklearn
	from sklearn.model_selection import train_test_split
	from sklearn.feature_extraction.text import CountVectorizer
	from sklearn.metrics import accuracy_score
	from sklearn.svm import LinearSVC

	#By removing stop words, we remove the low-level information from our text in order to give more focus to the important information.
	stop_words = ['in', 'of', 'at', 'a', 'the']

	ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 3), stop_words=stop_words)
	ngram_vectorizer.fit(reviews_train_clean)
	X = ngram_vectorizer.transform(reviews_train_clean)
	
	#Spliting the data set into training and validation set
	X_train, X_val, y_training, y_val = train_test_split(X, y_train, test_size=0.5)
	# Regularisation parameter optimises the model. So trying to find the right value of c to get most accuracy
	for c in [0.001, 0.005, 0.01, 0.05, 0.1]:
    
		svm = LinearSVC(C=c)
		svm.fit(X_train, y_training)
		print ("Accuracy for C=%s: %s" % (c, accuracy_score(y_val, svm.predict(X_val))))
			
	# Model with final selection of Regularization parameter   
	final = LinearSVC(C=0.01)
	final.fit(X_train, y_training)
	print ("Final Accuracy of training set: %s" % accuracy_score(y_train, final.predict(X)))
	# 3. Save your model
	
	file_name = "models/Group46_NLP_model.h5"
	pickle.dump(svm, open(file_name, 'wb'))
	pickle.dump(final, open(file_name, 'wb'))