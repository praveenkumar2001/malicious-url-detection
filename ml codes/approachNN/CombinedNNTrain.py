# Precondition : You already have a trained FetureLess model.

import csv
import pandas as pd
import numpy as np
import tensorflow as tf
from string import printable
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model, model_from_json, load_model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Input, ELU, Embedding, Convolution1D, MaxPooling1D,  Reshape, LSTM
from keras.optimizers import  Adam
from keras import backend as K
from keras import regularizers
from tensorflow.python.platform import gfile


# A function to freeze a Keras model and save it in .pb format
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.
    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph



# Catch words and catch chars, again taken from some reliable research paper
catchWords = ['secure', 'account', 'webser', 'login', 'ebayisapi', 'signin', 'login',
              'banking', 'confirm', 'blog', 'logon', 'signon', 'login.asp',
              'login.php', 'login.htm', '.exe', '.zip', '.rar', 'jpg', '.gif', 
              'viewer.php', 'link=', 'getImage.asp', 'plugins', 'paypal', 'order',
              'dbsys.php', 'config.bin', 'download.php', '.js', 'payment', 'files',
              'css', 'shopping', 'mail.php', '.jar', '.swf', '.cgi', '.php', 'abuse',
              'admin', 'ww1', 'personal', 'update', 'verification']
catchChars = [ '-', '_', '=',  '?', ';', '(', ')', '%', '&', '@']

# A function to extract features of a URL, take URL string as input
def features(s):
    data = []
    
    l = len(s)
    pos = s.find('?')
    q_len = 0
    if pos >-1:
        q_len = l-pos-1
    
    digits = sum(c.isdigit() for c in s)
    s1 = s
    s1.replace('.', '/')
    s1 = s1.split('/')
    token = len(s1)
    
    
    #data.append(l)
    #data.append(q_len)
    data.append(token)
    data.append(digits)
    
    
    for i in catchWords:
        if s.find(i)>-1:
            data.append(1)
        else:
            data.append(0)
            
    for i in catchChars:
        data.append((s.count(i))>0)
    return data

# A function to convert URL to a vector form	
def URL_integer(url):
    test_url = url
    test_url_int_tokens = [[printable.index(x) + 1 for x in test_url if x in printable] ]
    max_len=75
    test_X = sequence.pad_sequences(test_url_int_tokens, maxlen=max_len)
    return test_X   

# A function to predict a URL	
def fun_pred(wkdir, pb_filename, inputTensorName, outputTensorName, X_test):
    K.clear_session()
    with tf.Session() as sess:
        # load model from pb file
        with gfile.FastGFile(wkdir+'/'+pb_filename,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            g_in = tf.import_graph_def(graph_def)
        
        # inference by the model (op name must comes with :0 to specify the index of its output)
        tensor_output = sess.graph.get_tensor_by_name(outputTensorName)
        tensor_input = sess.graph.get_tensor_by_name(inputTensorName)
        predictions = sess.run(tensor_output,feed_dict =  {tensor_input : X_test})
    return predictions
	
	
# preparing the data fpr training the model
X = []
Y = []

safe = 0
unsafe = 0

with open('Dataset_Complete_Sorted_unique.txt', 'r') as csvfile: 
    i = 0
    csvreader = csv.reader(csvfile, delimiter='\t') 
    for row in csvreader:
        i = i+1
        link = row[0]
        positive = row[1]
        feature_vector = features(link)
        pred = fun_pred('./model/', 'featureLessModel_1DConvLSTM.pb', 'import/main_input:0', 'import/output/Sigmoid:0', URL_integer(link))
        feature_vector.append(pred[0][0])
        K.clear_session()
        
        if positive=='0':
            safe = safe+1
            Y.append([0])
            
        else:
            Y.append([1])
            unsafe = unsafe+1
        X.append(feature_vector)

# uncomment the line below to see the distribution of the dataset			
#print(safe, unsafe, safe+unsafe, len(X))

# solit the dara
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 9)

# uncomment the line below to see the split of trained test data
#print(len(X_train), len(Y_train), len(X_test), len(Y_test))

y_train = []
y_test = []
for i in range(0, len(Y_train)):
    y_train.append(Y_train[i][0])
    
for i in range(0, len(Y_test)):
    y_test.append(Y_test[i][0])

print(len(y_train), len(y_test))
print(len(X_train), len(X_test))

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# clear session and create a sequential model
K.clear_session()
combinedModel = Sequential()
combinedModel.add(Dense(56, activation = "relu", input_shape=(len(X[0]),), dtype='float'))
combinedModel.add(Dense(56, activation = "relu"))
combinedModel.add(Dense(28, activation = "relu"))
combinedModel.add(Dense(18, activation = "relu"))
combinedModel.add(Dense(1, activation = "sigmoid"))

#uncomment the line below to see model summary
#combinedModel.summary()

combinedModel.compile(optimizer = "adam", loss = "binary_crossentropy", metrics=["accuracy"])
results = combinedModel.fit(X_train, y_train, epochs = 100, batch_size = 10, validation_data = (X_test, y_test))

# freeze the graph so as to save that in .pb format
frozen_graph = freeze_session(K.get_session(),output_names=[out.op.name for out in combinedModel.outputs])
wkdir = './model'
pb_filename = 'CombinedNN_model.pb'
tf.train.write_graph(frozen_graph, wkdir, pb_filename, as_text=False)