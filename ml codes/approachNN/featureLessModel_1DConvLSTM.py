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


# Function to freeze the graph so as to save the model in a .pb format
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
        frozen_graph = convert_variables_to_constants(session, input_graph_def,output_names, freeze_var_names)
        return frozen_graph

def print_layers_dims(model):
    l_layers = model.layers
    # Note None is ALWAYS batch_size
    for i in range(len(l_layers)):
        print(l_layers[i])
        print('Input Shape: ', l_layers[i].input_shape, 'Output Shape: ', l_layers[i].output_shape)

def save_model(fileModelJSON,fileWeights):
    #print("Saving model to disk: ",fileModelJSON,"and",fileWeights)
    #have h5py installed
    if Path(fileModelJSON).is_file():
        os.remove(fileModelJSON)
    json_string = model.to_json()
    with open(fileModelJSON,'w' ) as f:
        json.dump(json_string, f)
    if Path(fileWeights).is_file():
        os.remove(fileWeights)
    model.save_weights(fileWeights)
    
def load_model(fileModelJSON,fileWeights):
    #print("Saving model to disk: ",fileModelJSON,"and",fileWeights)
    with open(fileModelJSON, 'r') as f:
         model_json = json.load(f)
         model = model_from_json(model_json)
    
    model.load_weights(fileWeights)
    return model


# The main lstm_conv function, to create a complete 2D Conv architecture for
# prediction of URL being safe or unsafe
def lstm_conv(max_len=75, emb_dim=35
              , max_vocab_len=100, lstm_output_size=32, W_reg=regularizers.l2(1e-4)):
    # Input
    main_input = Input(shape=(max_len,), dtype='float', name='main_input')
    # Embedding layer
    emb = Embedding(input_dim=max_vocab_len, output_dim=emb_dim, input_length=max_len,
                W_regularizer=W_reg)(main_input) 
    emb = Dropout(0.25)(emb)

    # Conv layer
    conv = Convolution1D(kernel_size=5, filters=256, \
                     border_mode='same')(emb)
    conv = ELU()(conv)

    conv = MaxPooling1D(pool_size=4)(conv)
    conv = Dropout(0.5)(conv)

    # LSTM layer
    lstm = LSTM(lstm_output_size)(conv)
    lstm = Dropout(0.5)(lstm)
    
    # Output layer (last fully connected layer)
    output = Dense(1, activation='sigmoid', name='output')(lstm)

    # Compile model and define optimizer
    model = Model(input=[main_input], output=[output])
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Read the data from the file	
fileName = 'Dataset_Complete_Sorted_unique.txt' 
df = pd.read_csv(fileName, sep='\t', header=None)
df.columns = ['url', 'positives', 'total']
url_int_tokens = [[printable.index(x) + 1 for x in url if x in printable] for url in df.url]
max_len=75
X = sequence.pad_sequences(url_int_tokens, maxlen=max_len)
target = np.array(df.positives)
for i in range(0, len(target)):
    if target[i]>0:
        target[i]=1

# Uncomment the below line to check the dimensions of the matrix
#print('Matrix dimensions of X: ', X.shape, 'Vector dimension of target: ', target.shape)

# perform a train test split for training the model
X_train, X_test, target_train, target_test = train_test_split(X, target, test_size=0.25, random_state=33)

model = lstm_conv()
#print(model.summary())


# set the epochs and the batch size, then train the model
epochs = 100
batch_size = 30
model.fit(X_train, target_train, epochs=epochs, batch_size = 32, validation_data = (X_test, target_test))

# analyze the performance of the model
loss, accuracy = model.evaluate(X_test, target_test, verbose=1)
print(loss, accuracy)


# freezing the graph and saving the model
frozen_graph = freeze_session(K.get_session(),output_names=[out.op.name for out in model.outputs])
wkdir = './model'
pb_filename = 'featureLessModel_1DConvLSTM.pb'
tf.train.write_graph(frozen_graph, wkdir, pb_filename, as_text=False)