import csv
import pandas as pd
import numpy as np
import sys
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

combinedModelName = 'GiveSomeSensibleNameHere_combinedModel.pb'
featuresLess_LSTM_CNN_model = 'lstm_convmodel_Dataset_Complete_Sorted_unique.pb'

catchWords = ['secure', 'account', 'webser', 'login', 'ebayisapi', 'signin', 'login',
              'banking', 'confirm', 'blog', 'logon', 'signon', 'login.asp',
              'login.php', 'login.htm', '.exe', '.zip', '.rar', 'jpg', '.gif', 
              'viewer.php', 'link=', 'getImage.asp', 'plugins', 'paypal', 'order',
              'dbsys.php', 'config.bin', 'download.php', '.js', 'payment', 'files',
              'css', 'shopping', 'mail.php', '.jar', '.swf', '.cgi', '.php', 'abuse',
              'admin', 'ww1', 'personal', 'update', 'verification']

#atchChars = ['.', '-', '_', '=', '/', '?', ';', '(', ')', '%', '&', '@']
catchChars = [ '-', '_', '=',  '?', ';', '(', ')', '%', '&', '@']

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


def URL_integer(url):
    test_url = url
    test_url_int_tokens = [[printable.index(x) + 1 for x in test_url if x in printable] ]
    max_len=75
    test_X = sequence.pad_sequences(test_url_int_tokens, maxlen=max_len)
    return test_X   



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
        #tensor_input = X_test
        #predictions = sess.run(tensor_output,feed_dict =  {tensor_input : X_test})
        predictions = sess.run(tensor_output,feed_dict =  {tensor_input : X_test})
        #print('===== output predicted results =====\n')
        #print(predictions)  
    return predictions
	
files = [featuresLess_LSTM_CNN_model , combinedModelName]
in_names = ['import/main_input:0','import/dense_1_input:0' ]
op_names = ['import/output/Sigmoid:0', 'import/dense_5/Sigmoid:0']
link =sys.argv[1]

def fun_pred_combined(wkdir, pb_filename, inputTensorName, outputTensorName, link):
    K.clear_session()
    pred_featureless = 0
    pred_combined = 0
    with tf.Session() as sess:
        # load model from pb file
        X_test = URL_integer(link)
        with gfile.FastGFile(wkdir+'/'+pb_filename[0],'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            g_in = tf.import_graph_def(graph_def)
        
        # inference by the model (op name must comes with :0 to specify the index of its output)
        tensor_output = sess.graph.get_tensor_by_name(outputTensorName[0])
        tensor_input = sess.graph.get_tensor_by_name(inputTensorName[0])
        
        pred_featureless = sess.run(tensor_output,feed_dict =  {tensor_input : X_test})
    
    K.clear_session()
    with tf.Session() as sess:
        # load model from pb file
        feature_vector = features(link)
        feature_vector.append(pred_featureless[0][0])
        feature_vector = [np.array(feature_vector)]
        feature_vector = np.array(feature_vector)
        X_test = feature_vector
        
        with gfile.FastGFile(wkdir+'/'+pb_filename[1],'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            g_in = tf.import_graph_def(graph_def)
        
        # inference by the model (op name must comes with :0 to specify the index of its output)
        tensor_output = sess.graph.get_tensor_by_name(outputTensorName[1])
        tensor_input = sess.graph.get_tensor_by_name(inputTensorName[1])
        pred_combined = sess.run(tensor_output,feed_dict =  {tensor_input : X_test})
    return pred_featureless, pred_combined
	
K.clear_session()
sol = fun_pred_combined('./model/', files, in_names, op_names, link)
print(sol)