import csv
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense
from keras import backend as K


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



if __name__=="__main__":
	X = []
	Y = []

	with open('outputFile.txt', 'r') as f:
		csvreader = csv.reader(f, delimiter='\t') 
		for row in csvreader:
			prop = row[1:-1]
			label = row[-1]
			X.append(prop)
			Y.append(label)

	X[0]

	#split data
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 9)

	X_train = np.array(X_train)
	Y_train = np.array(Y_train)
	X_test = np.array(X_test)
	Y_test = np.array(Y_test)

	model = Sequential()

	model.add(Dense(6, activation = "relu", input_shape=(6,), dtype='float'))
	model.add(Dense(12, activation = "relu"))
	model.add(Dense(12, activation = "relu"))
	model.add(Dense(6, activation = "relu"))
	model.add(Dense(1, activation = "sigmoid"))
	model.summary()

	model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics=["accuracy"])

	results = model.fit(X_train, Y_train, epochs = 40, batch_size = 3, validation_data = (X_test, Y_test))


	frozen_graph = freeze_session(K.get_session(),
								  output_names=[out.op.name for out in model.outputs])
	wkdir = './model'
	pb_filename = 'NN_model_finalOutput.pb'
	tf.train.write_graph(frozen_graph, wkdir, pb_filename, as_text=False)