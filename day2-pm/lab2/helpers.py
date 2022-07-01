# This file contains codes abstracted away from the main Jupyter Notebook.

# Import libraries
#
import pandas as pd
import numpy as np
import os
from datetime import datetime
import pickle

import tensorflow
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Bidirectional, Embedding, Dropout, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.compat.v1.keras.layers import CuDNNLSTM

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from functools import reduce
import pickle

# Declare global ariables used by the abstracted functions
#
text_classifier_model = None
text_classifier_labels = None
text_tokenizer = None

glove_embedding_matrix = None

train_x = None
train_y = None
test_x = None
test_y = None

history = None

print("version 1.2 loaded")
# Loads all text data from the CSV file into memory#
#
def load_text_data_from_csv(train_csv_file, test_csv_file, text_column, class_column):
    global text_tokenizer, train_x, train_y, test_x, test_y, glove_embedding_matrix, text_classifier_labels
    
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
    text_classifier_labels = np.unique(combined_y).tolist()
    
    unique_classes_map = {}
    for i, classname in enumerate(text_classifier_labels):
        unique_classes_map[classname] = i
        
    # Convert the 'Y' data (labels) into the format for Keras
    #
    train_y = [unique_classes_map[c] for c in train_y]
    train_y = to_categorical(train_y)
    test_y = [unique_classes_map[c] for c in test_y]
    test_y = to_categorical(test_y)
    
    print ("Complete.")
    

# Tokenize the loaded data into individual word indexes.
# This function must run IMMEDIATELY AFTER 
# load_text_data_from_csv.
#
def build_dictionary_and_tokenize_data(max_words=30000, max_sentence_length=250):
    global text_tokenizer, train_x, test_x
    
    # Initialize the Keras tokenizer.
    # The tokenizer helps us to convert sentences (sequences of words) into 
    # into sequences of numeric values that can be fed into the Word Embedding 
    # layer.
    # 
    print ("Initializing tokenizer...")
    train_and_test_x = np.concatenate((train_x, test_x))
    text_tokenizer = Tokenizer(num_words=max_words, lower=True, split=" ")
    text_tokenizer.fit_on_texts(train_and_test_x)
    
    # Tokenize all training input sentences
    #
    print ("Tokenizing training data...")
    train_x = text_tokenizer.texts_to_sequences(train_x)
    max_x = 0
    for i in range(0, len(train_x)):
        if (len(train_x[i]) > max_x):
            max_x = len(train_x[i])
    print ("  Max number of words in a sentence: %d" % (max_x))
    train_x = np.array(pad_sequences(train_x, maxlen=max_sentence_length))
    
    # Tokenize all testing input sentences
    #
    print ("Tokenizing test data...")
    test_x = text_tokenizer.texts_to_sequences(test_x)
    max_x = 0
    for i in range(0, len(test_x)):
        if (len(test_x[i]) > max_x):
            max_x = len(test_x[i])
    print ("  Max number of words in a sentence: %d" % (max_x))
    test_x = np.array(pad_sequences(test_x, maxlen=max_sentence_length))
    
    print ("Complete.")
     
        
# Loads the Glove embeddings
#
def load_glove_embedding(file):

    global glove_embedding_matrix, text_tokenizer
    
    # Load up the GloVe word embedding data
    #
    print("Loading GloVe Word Embedding...")
    embeddings_index = {}
    f = open(file, encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    
    # Construct the word embedding matrix that will be used in the 
    # Embedding layer.
    #
    word_index = text_tokenizer.word_index
    glove_embedding_matrix = np.zeros((len(word_index) + 1, 200))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            glove_embedding_matrix[i] = embedding_vector
    
    print ("Complete.")

# Gets the embedding of a word for testing.
#
def display_word_embedding(word):
    word_index = text_tokenizer.word_index[word]
    
    print ("Index of word %s: %d" % (word, word_index))
    print ("Word embedding:")
    print (glove_embedding_matrix[word_index])
    

# Displays a list of words that are nearby the given word
#
def display_nearby_words(word, number_of_nearby_words=10):
    word_index = text_tokenizer.word_index[word]
    embedding = glove_embedding_matrix[word_index]
    
    [(i, np.linalg.norm(glove_embedding_matrix[i] - embedding)) for i in range(0, glove_embedding_matrix.shape[0])]

    
# Displays a list of words that are nearby the given word
#
def display_nearby_words(word, number_of_nearby_words=10):
    word_index = text_tokenizer.word_index[word]
    embedding = glove_embedding_matrix[word_index]
    
    dist = np.array([[int(i), np.linalg.norm(glove_embedding_matrix[i] - embedding)] for i in range(0, glove_embedding_matrix.shape[0])])
    index = dist[:, 1].argsort()
    
    reverse_word_map = dict(map(reversed, text_tokenizer.word_index.items()))

    for i in range(1,number_of_nearby_words + 1):
        if index[i] != 0:
            print ("%-30s %f" % (reverse_word_map[index[i]], dist[index[i]][1]))
        
        
# Displays train_x and train_y
#
def display_trainx_trainy():
    print (train_x)
    print (train_y)

# Displays test_x and test_y
#
def display_testx_testy():
    print (test_x)
    print (test_y)

# Create a new Keras text classifier with the following architecture:
#    Word Embedding 
#    RNN (either RNN, GRU or LSTM)
#    Dense (also known as Fully Connected Layer)
#
def create_text_classifier_model_rnn(
    number_of_classes,
    max_sequence_length=250,
    embedding='glove', 
    rnn_type='rnn', 
    rnn_units=128, 
    use_bidirectional=False,
    optimizer='sgd'):
    
    global text_classifier_model
    global glove_embedding_matrix
    
    text_classifier_model = Sequential()
    
    # Add an Embedding layer
    #
    if embedding == 'glove':
        text_classifier_model.add(
            Embedding(glove_embedding_matrix.shape[0], glove_embedding_matrix.shape[1], 
                      weights=[glove_embedding_matrix], input_length=max_sequence_length, trainable=False))
    else:
        text_classifier_model.add(
            Embedding(input_dim=len(text_tokenizer.word_index) + 1, output_dim=200, 
                      input_length=max_sequence_length))

    # Add a RNN (or one of its variants)
    #
    if rnn_type == 'lstm':
        if use_bidirectional:
            text_classifier_model.add(Bidirectional(LSTM(rnn_units)))
            text_classifier_model.add(Activation("relu"))
        else:
            text_classifier_model.add(LSTM(rnn_units))
            text_classifier_model.add(Activation("relu"))
            
    if rnn_type == 'gru':
        if use_bidirectional:
            text_classifier_model.add(Bidirectional(GRU(rnn_units)))
            text_classifier_model.add(Activation("relu"))
        else:
            text_classifier_model.add(GRU())
            text_classifier_model.add(Activation("relu"))
            
    if rnn_type == 'rnn':
        if use_bidirectional:
            text_classifier_model.add(Bidirectional(SimpleRNN(rnn_units, activation='relu')))
        else:
            text_classifier_model.add(SimpleRNN(rnn_units, activation='relu'))

    # Add a Dense layer with 2 units.
    #
    text_classifier_model.add(Dense(number_of_classes, activation='softmax'))

    # Print the model summary
    #
    text_classifier_model.summary()
    
    # Compile the model with the loss and optimizer functions
    #
    text_classifier_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    


# This function is used to display the history the train/test accuracy/loss
# of the Keras training.
#
#   history - Pass in the history returned from the model.fit(...) method.
#
def display_training_loss_and_accuracy(history):
    
    plt.figure(figsize=(20,4))
    
    # summarize history for accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    
    # summarize history for loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()    

# Display evaluation results
#
def display_model_evaluation_results(y_test, pred_y_test, labels):
    
    plt.figure(figsize=(20,6))  

    labels = np.array(labels)      
    
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
    print ("Test Data")
    print ("--------------------------------------------------------")
    print(classification_report(y_test, pred_y_test, target_names=labels))
    

# Set the learning rate.
#
def set_learning_rate(lr):
    global text_classifier_model
    from tensorflow.keras import backend as K
    K.set_value(text_classifier_model.optimizer.learning_rate, lr)


# Train the text classifier model
#
def train_text_classifier_model(batch_size=8, epochs=20):
    global text_classifier_model
    global history

    '''   
    # Create the training folder
    #
    training_session_id = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    training_session_folder = folder + '/train_%s' % (training_session_id)
    os.makedirs(training_session_folder, exist_ok=True)

    # Configure the checkpoint and stop point.
    # This allows the training to save the best models and also stop the
    # training early if it detects that there are no improvements after
    # a long time.
    #
    callbacks_list = [
        ModelCheckpoint(
            filepath=training_session_folder + '/model.{epoch:04d}-acc-{accuracy:4.2f}-val_acc-{val_accuracy:4.2f}-loss-{val_loss:4.2f}.h5',
            monitor='val_loss', save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=100)
    ]
    '''

    # Perform the training.
    #
    history = text_classifier_model.fit(train_x, train_y, validation_data=(test_x, test_y), batch_size=batch_size, epochs=epochs)
    
    # save the result and model 
    with open('train_history.pkl', 'wb') as file_pi:
        pickle.dump(history, file_pi)
    
    text_classifier_model.save("sentiment.model")

    # display_training_loss_and_accuracy(history)
    # display_model_evaluation_results(actual_train_y, pred_train_y, actual_test_y, pred_test_y, text_classifier_labels)

def display_training_progress(): 
    display_training_loss_and_accuracy(history)

def evaluate_model(): 
    global text_classifier_model
    # Evaluate and display the results of the trained model.
    #
    print ("Evaluating classifier...")
    pred_test_y = text_classifier_model.predict(test_x)

    actual_test_y = np.argmax(test_y, axis=1)
    pred_test_y = np.argmax(pred_test_y, axis=1)

    display_model_evaluation_results(actual_test_y, pred_test_y, text_classifier_labels)

def load_pretrained(): 
    global text_classifier_model
    global history 
    
    text_classifier_model = keras.models.load_model('sentiment.model')
    history = pickle.load(open("train_history.pkl", "rb"))

# Save the Text Classifier model
#
def save_text_classifier_model(filename):
    global text_classifier_model, text_classifier_labels, text_tokenizer
    pickle.dump(text_tokenizer, open(filename + ".tokenizer", "wb"))
    pickle.dump(text_classifier_labels, open(filename + ".labels", "wb"))
    text_classifier_model.save(filename)
    
    
# Load the Text Classifier model
#
def load_text_classifier_model(filename):
    global text_classifier_model, text_classifier_labels, text_tokenizer
    text_tokenizer = pickle.load(open(filename + ".tokenizer", "rb"))
    text_classifier_labels = pickle.load(open(filename + ".labels", "rb"))
    text_classifier_model = tensorflow.keras.models.load_model(filename)
    
# Classify the Text Classifier model
#    
def classify_text(text):
    print ("You entered: %s" % (text))

    max_sentence_length = text_classifier_model.layers[0].input.shape[1]
    text = np.array(text_tokenizer.texts_to_sequences([text]))
    text = pad_sequences(text, maxlen=max_sentence_length)

    prediction = text_classifier_model.predict(text)
    
    print ("Classification result:")    
    print (text_classifier_labels[np.argmax(prediction, axis=1)[0]])

    for i in range(0, len(text_classifier_labels)):
        print ("%5.3f - %s" % (prediction[0][i], text_classifier_labels[i]))