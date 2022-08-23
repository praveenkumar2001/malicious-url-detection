import csv
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense
from keras import backend as K

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
    
    
    data.append(l)
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


# reading the URLs from file and creating the training and testing data
X = []
Y = []

safe = 0
unsafe = 0
fileName = 'Dataset_Complete_Sorted_unique.txt'

# read from some file
with open(fileName, 'r') as csvfile: 
    i = 0
    csvreader = csv.reader(csvfile, delimiter='\t') 
    for row in csvreader:
        i = i+1
        link = row[0]
        positive = row[1]
        feature_vector = features(link)
        if positive=='0':
            safe = safe+1
            Y.append(0)
            
        else:
            Y.append(1)
            unsafe = unsafe+1
           
        X.append(feature_vector)
        

# a print statement to analyze data distribution, this is not copied ;)
print(safe, unsafe, safe+unsafe, len(X))

# split data using train-test split, and convert them to np.arrays
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 9)
X_train = np.array(X_train)
X_test = np.array(X_test)


# create a sequential Keras model
model = Sequential()
model.add(Dense(56, activation = "relu", input_shape=(len(X_train[0]),), dtype='float'))
model.add(Dense(56, activation = "relu"))
model.add(Dense(28, activation = "relu"))
model.add(Dense(18, activation = "relu"))
model.add(Dense(1, activation = "sigmoid"))

# uncomment the statement below to print model.summary()
# print(model.summary())

# compile the model and check the results
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics=["accuracy"])
results = model.fit(X_train, Y_train, epochs = 21, batch_size = 3, validation_data = (X_test, Y_test))

print(results)


# freeze the graph so as to save that in .pb format
frozen_graph = freeze_session(K.get_session(),output_names=[out.op.name for out in model.outputs])
wkdir = './model'
pb_filename = 'featureBasedModel .pb'
tf.train.write_graph(frozen_graph, wkdir, pb_filename, as_text=False)
