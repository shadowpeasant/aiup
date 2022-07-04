# Import necessary libraries
#
import pandas as pd
import numpy as np
import os
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn import metrics

import nltk
from nltk import word_tokenize   
from nltk.stem import WordNetLemmatizer 

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from functools import reduce


# Declare global variables required for the model and data.
#
text_classifier_model = None
text_classifier_labels = None
train_x = None
train_y = None
test_x = None
test_y = None


# Tokenizer for processing the words
#
class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
    

# Create a Naive-Bayes classifier using Scikit Learn.
#
# The processing pipeline performs the following:
# - Lemmatization
# - Bag-of-Words (CountVectorizer)
# - TF-IDF
# - Naive Bayes model
#
def create_text_classifier_model_naivebayes(alpha=1.0, fit_prior=True, class_prior=None):
    global text_classifier_model
    
    text_classifier_model = Pipeline([
        ('vect', CountVectorizer(tokenizer=LemmaTokenizer())),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB(alpha=alpha, fit_prior=fit_prior, class_prior=None)),
    ])
    
# Create a SVM classifier using Scikit Learn
#
# The processing pipeline performs the following:
# - Lemmatization
# - Bag-of-Words (CountVectorizer)
# - TF-IDF
# - SVM model
#
def create_text_classifier_model_svm():
    global text_classifier_model
    text_classifier_model = Pipeline([
        ('vect', CountVectorizer(tokenizer=LemmaTokenizer())),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(verbose=1) ),
    ])        
    
#
# The processing pipeline performs the following:
# - Lemmatization
# - Bag-of-Words (CountVectorizer)
# - TF-IDF
# - Multi-Layer Perceptron (Artificial Neural Network)
#
def create_text_classifier_model_ann(hidden_layer_sizes=(50, ), epochs=10):
    global text_classifier_model
    text_classifier_model = Pipeline([
        ('vect', CountVectorizer(tokenizer=LemmaTokenizer())),
        ('tfidf', TfidfTransformer()),
        ('clf', MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=hidden_layer_sizes, random_state=1, max_iter=epochs, verbose=True) ),
    ])        
          

# Load the training and test data from CSV that can be passed into Scikit
# Learn for training and evaluation.
#
def load_text_data_from_csv_for_scikit(train_csv_file, test_csv_file, text_column, class_column):
    global text_classifier_model, train_x, train_y, test_x, test_y, text_classifier_labels
    
    # Load up the data from the training CSV file.
    #
    print ("Loading training data...")
    train_data_df = pd.read_csv(train_csv_file)
    train_x = train_data_df[[text_column]].to_numpy()
    train_x = train_x.reshape((train_x.shape[0]))
    train_y = train_data_df[[class_column]].to_numpy()
    train_y = train_y.reshape((train_y.shape[0]))
    
    # Load up the data from the test CSV file.
    #    
    print ("Loading test data...")
    test_data_df = pd.read_csv(test_csv_file)
    test_x = test_data_df[[text_column]].to_numpy()
    test_x = test_x.reshape((test_x.shape[0]))
    test_y = test_data_df[[class_column]].to_numpy()
    test_y = test_y.reshape((test_y.shape[0]))
    
    # Create a combined unique list of labels
    # 
    combined_y = np.concatenate((train_y, test_y)).reshape((train_y.shape[0] + test_y.shape[0], 1))
    unique_classes = np.unique(combined_y)
    text_classifier_labels = unique_classes.tolist()
    
    unique_classes_map = {}
    for i, classname in enumerate(unique_classes.tolist()):
        unique_classes_map[classname] = i
    
    # Convert the 'Y' data (labels) into the format for Scikit-Learn
    #
    train_y = np.array([unique_classes_map[classname] for classname in train_y.tolist()])
    test_y = np.array([unique_classes_map[classname] for classname in test_y.tolist()])
    print ("Loading complete.")

# Display evaluation results
#
def display_model_evaluation_results(y_train, pred_y_train, y_test, pred_y_test, labels):
    
    plt.figure(figsize=(20,6))  

    labels = np.array(labels)
 
    # Print the first Confusion Matrix for the training data
    #
    cm = confusion_matrix(y_train, pred_y_train)

    cm_df = pd.DataFrame(cm, labels, labels)          
    plt.subplot(1, 2, 1)
    plt.title('Confusion Matrix (Train Data)')
    sns.heatmap(cm_df, annot=True)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')        
    
    # Print the second Confusion Matrix for the test data
    #    
    cm = confusion_matrix(y_test, pred_y_test)
    
    cm_df = pd.DataFrame(cm, labels, labels)          
    plt.subplot(1, 2, 2)
    plt.title('Confusion Matrix (Test Data)')
    sns.heatmap(cm_df, annot=True)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')        
    
    plt.show()

    # Finally display the classification reports
    #
    print ("Train Data")
    print ("--------------------------------------------------------")
    print(metrics.classification_report(y_train, pred_y_train, target_names=labels))
    print ("")
    print ("Test Data")
    print ("--------------------------------------------------------")
    print(metrics.classification_report(y_test, pred_y_test, target_names=labels))

# Display the train_X and train_Y data.
#
def display_trainx_trainy():
    print (train_x)
    print (train_y)

# Display the test_X and test_Y data.
#
def display_testx_testy():
    print (test_x)
    print (test_y)


# Train the text classifier model
#
def train_text_classifier_model():
    print ("Training classifier...")
    text_classifier_model.fit(train_x, train_y)
    
    print ("Evaluating classifier...")
    pred_y_train = text_classifier_model.predict(train_x)
    pred_y_test = text_classifier_model.predict(test_x)
    
    display_model_evaluation_results(train_y, pred_y_train, test_y, pred_y_test, text_classifier_labels)

def save_text_classifier_model(filename):
    global text_classifier_model, text_classifier_labels
    pickle.dump(text_classifier_labels, open(filename + ".labels", "wb"))
    pickle.dump(text_classifier_model, open(filename, "wb"))
    
    
def load_text_classifier_model(filename):
    global text_classifier_model, text_classifier_labels
    text_classifier_labels = pickle.load(open(filename + ".labels", "rb"))
    text_classifier_model = pickle.load(open(filename, "rb"))
    
    
def classify_text(text):
    print ("You entered: %s" % (text))
    result = text_classifier_model.predict([text])

    print ("Classification result:")
    print (text_classifier_labels[result[0]])

nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('wordnet')