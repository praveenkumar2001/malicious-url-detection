# Authors : Github, StackOverflow, some more github, some more stackOverflow
# Precondition : You already have a trained FetureLess model, trained featureBased model.
# 				 ML models trained on certificate and nw flow attributes parameter

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


#reading from the command line
import sys
import pickle
#establising connection and downloading the certificate
import datetime
#reading the certificate
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography import utils
from cryptography.x509.oid import NameOID, ObjectIdentifier
from cryptography.x509.oid import ExtensionOID
from cryptography.hazmat.primitives import hashes


import os
import signal
import subprocess as sub
#import wget
import requests
import time
import sys 
import socket as socket
import ssl as ssl
from statistics import mean,stdev


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

#NN stuff right here
combinedModelName = 'CombinedNN_model.pb'
featuresLess_LSTM_CNN_model = 'featureLessModel_1DConvLSTM.pb'
files = [featuresLess_LSTM_CNN_model , combinedModelName]
in_names = ['import/main_input:0','import/dense_1_input:0' ]
op_names = ['import/output/Sigmoid:0', 'import/dense_5/Sigmoid:0']

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
    return pred_featureless[0][0], pred_combined[0][0]

	
# A function to download certificate for a URL	
def get_certificate(url):
	hostname = url
	port = 443
	
	try:
		
		context = ssl.create_default_context()
		conn = ssl.create_connection((hostname, port),timeout = 0.5)
		sock = context.wrap_socket(conn, server_hostname  = hostname)
		der_cert = sock.getpeercert(True)
		return 1, der_cert
	except Exception as e:
		print(e)
		try:
			pem_cert = ssl.get_server_certificate((hostname, port))
			der_cert = ssl.PEM_cert_to_DER_cert(pem_cert)
			return 1, der_cert
		except Exception as e:
			print(e)
			return 0, 0

# A function to extract properties from the certificate
def extract_properties_cert(cert):
	#cert = x509.load_der_x509_certificate(derCert, default_backend())
	properties = []
	try:
		key_usage_object = cert.extensions.get_extension_for_oid(ExtensionOID.KEY_USAGE)
		key_usage_critical = int(key_usage_object._critical)
		digital_signature = int(key_usage_object._value._digital_signature)
		content_commitment = int(key_usage_object._value._content_commitment)
		key_encipherment = int(key_usage_object._value._key_encipherment)
		data_encipherment = int(key_usage_object._value._data_encipherment)
		key_agreement = int(key_usage_object._value._key_agreement)
		key_cert_sign = int(key_usage_object._value._key_cert_sign)
		crl_sign = int(key_usage_object._value._crl_sign)
		encipher_only = int(key_usage_object._value._encipher_only)
		decipher_only = int(key_usage_object._value._decipher_only)
			
	except:
		key_usage_critical = 0
		digital_signature = 0
		content_commitment = 0
		key_encipherment = 0
		data_encipherment = 0
		key_agreement = 0
		key_cert_sign = 0
		crl_sign = 0
		encipher_only = 0
		decipher_only = 0
		

	#getting the info from from_date and to_date parameters
	
	from_date =  cert.not_valid_before
	to_date =  cert.not_valid_after
	curr_date = datetime.datetime.now()
	days_before = curr_date-from_date
	days_before = (days_before.days)
	days_after = to_date-curr_date
	days_after = (days_after.days)
	
	if days_before>60:
		days_before = 2
	elif days_before>30:
		days_before = 1
	elif days_before>0:
		days_before = 0
	elif days_before>-30:
		days_before = -1
	elif days_before>-60:
		days_before = -2
	else:
		days_before = -3
	
	
	if days_after>60:
		days_after = 2
	elif days_after>30:
		days_after = 1
	elif days_after>0:
		days_after = 0
	elif days_after>-30:
		days_after = -1
	elif days_after>-60:
		days_after = -2
	else:
		days_after = -3
	
	#getting the basic constraints
	try:
		basic_constraints_object = cert.extensions.get_extension_for_oid(ExtensionOID.BASIC_CONSTRAINTS)
		basic_constarints_critical = int(basic_constraints_object._critical)
		ca = int(basic_constraints_object._value._ca)
	except:
		basic_constarints_critical = 0
		ca = 0
	
	#getting the host name feilds from subject
	cn = ""
	try:
		subject = cert.subject
		cn_object = subject.get_attributes_for_oid(NameOID.COMMON_NAME)
		for k in cn_object:
			cn = str(k._value)
	except:
		cn=""
	
	#getting the issuer
	cn_issuer = ""
	try:
		issuer = cert.issuer
		cn_object_issuer = issuer.get_attributes_for_oid(NameOID.COMMON_NAME)
		for k in cn_object_issuer:
			cn_issuer = str(k._value)
	except:
		cn_issuer=""
	
	self_signed = 0
	if cn_issuer==cn:
		self_signed = 1

	#getting the signature hash algorithm used in the certificate
	algorithm=[0]*16
	try:
		algorithm_object = cert.signature_algorithm_oid
		algorithm_name = algorithm_object._name
		algorithm_dotted_string = algorithm_object._dotted_string
		
		if algorithm_dotted_string == "1.2.840.113549.1.1.4":
			algorithm[0] = 1 #RSA_MD5
		elif algorithm_dotted_string == "1.2.840.113549.1.1.5":
			algorithm[1] = 1 #RSA_SHA1
		elif algorithm_dotted_string == "1.3.14.3.2.9":
			algorithm[2] = 1 #RSA_SHA1_
		elif algorithm_dotted_string == "1.2.840.113549.1.1.14":
			algorithm[3] = 1 #RSA_SHA224
		elif algorithm_dotted_string == "1.2.840.113549.1.1.11":
			algorithm[4] = 1 #RSA_SHA256
		elif algorithm_dotted_string == "1.2.840.113549.1.1.12":
			algorithm[5] = 1 #RSA_SHA384
		elif algorithm_dotted_string == "1.2.840.113549.1.1.13":
			algorithm[6] = 1 #RSA_SHA512
		elif algorithm_dotted_string == "1.2.840.113549.1.1.10":
			algorithm[7] = 1 #RSA_SSA_PSS
		elif algorithm_dotted_string == "1.2.840.10045.4.1":
			algorithm[8] = 1 #ECDSA_SHA1
		elif algorithm_dotted_string == "1.2.840.10045.4.3.1":
			algorithm[9] = 1 #ECDSA_SHA224
		elif algorithm_dotted_string == "1.2.840.10045.4.3.2":
			algorithm[10] = 1 #ECDSA_SHA256
		elif algorithm_dotted_string == "1.2.840.10045.4.3.3":
			algorithm[11] = 1 #ECDSA_SHA384
		elif algorithm_dotted_string == "1.2.840.10045.4.3.4":
			algorithm[12] = 1 #ECDSA_SHA512
		elif algorithm_dotted_string == "1.2.840.10040.4.3":
			algorithm[13] = 1 #DSA_SHA1
		elif algorithm_dotted_string == "2.16.840.1.101.3.4.3.1":
			algorithm[14] = 1 #DSA_SHA224
		elif algorithm_dotted_string == "2.16.840.1.101.3.4.3.2":
			algorithm[15] = 1 #DSA_SHA256
	except:
		algorithm_name = "Unknown OID"
		
	
	#getting the key size
	key_size = 0
	try:
		key_size = (cert.public_key().key_size)/128
		key_size = int(key_size)
	except:
		key_size = 0
		
	properties = [key_usage_critical ,digital_signature ,content_commitment , key_encipherment , data_encipherment , key_agreement , key_cert_sign , crl_sign , encipher_only , decipher_only, days_before, days_after, basic_constarints_critical, ca, key_size, self_signed]
	properties = properties + algorithm
	return properties

# A function to run a ML model on cert properties
def predict_certificate_prop(data, model_path):
	#model_path = 'certificate_classifier_model.sav'
	model = pickle.load(open(model_path, 'rb'))
	return model.predict_proba(data)[0]	

# A function to extract the nw flow parameters for a given URL
def extract_flow_data(url):
	process1 = sub.Popen(["nslookup", url], stdout=sub.PIPE)
	output = process1.communicate()[0].decode('utf-8').split('\n')
		
	error='no error'
	ip_arr = []
	zeros=[0]*15
	zeros.append('0')
	for data in output:
		if 'Address' in data:
			ip_arr.append(data.replace('Address: ',''))	
	ip_arr.pop(0)

	#print(ip_arr[0])
	if len(ip_arr)==0:
		##print('DNS not found')
		zeros.append('DNS not found')
		return zeros

	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	ssl_sock = ssl.wrap_socket(s,cert_reqs=ssl.CERT_REQUIRED,ca_certs='/etc/ssl/certs/ca-certificates.crt')
	try:
		ssl_sock.connect((ip_arr[0], 443))
		cipher=ssl_sock.cipher()[0]
	except Exception as e:
		error=e
		cipher=0
		
	#cmd='tcpdump -w 001.pcap -i eth0 tcp and src 172.217.27.196 or dst 172.217.27.196'
	process2 = sub.Popen(('tcpdump', '-w', '001.pcap', '-i', 'eth0','-v', 'tcp', 'and', 'src', ip_arr[0], 'or','dst',ip_arr[0]), stdout=sub.PIPE)

	url='https://'+ url
	try:
		r = requests.get(url, timeout=10)
	except Exception as e:
		zeros.append('requests error')
		process2.terminate()
		return zeros
	time.sleep(2)
	process2.terminate()
	#os.killpg(os.getpgid(p.pid), signal.SIGTERM)  # Send the signal to all the process groups
	
	
	cmd='tshark -r 001.pcap -T fields -e _ws.col.Time -e _ws.col.Source -e _ws.col.Destination -e _ws.col.SrcPort -e _ws.col.DstPort -e  _ws.col.Protocol -e _ws.col.Length -E separator=, -E occurrence=f > pcap_traffic.csv'
	process3 = sub.Popen(cmd,shell=True,stdout=sub.PIPE)
	time.sleep(1)
	process3.terminate()
	

	arr=[]
	in_bytes=0
	out_bytes=0
	in_packets=0
	out_packets=0
	ip='192.168.1.4'
	i=0
	src_port=0
	dst_port=0
	total_time=0
	packet_lengths=[]
	inter_arrival_time=[]
	i=0
	temp=0;
	z=0
	with open('pcap_traffic.csv') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		for row in csv_reader:
			if int(row[6])==0:
				break
			if z==0 and row[1]==ip_arr[0]:
				src_port=8001
				dst_port=8001
				z=1
			if i>0:
				inter_arrival_time.append(float(row[0])-temp)
				temp=float(row[0])
			packet_lengths.append(int(row[6]))
			if row[6]=='0':
				break
			i+=1
			total_time=row[0]
			if row[1]==ip:
				out_bytes+=int(row[6])
				out_packets+=1
			else:
				in_bytes+=int(row[6])
				in_packets+=1
	if i==0:
		zeros.append('no packets captured')
		return zeros
	
		
	#print(in_packets)
	#print(out_packets)
	try:
		std1=stdev(inter_arrival_time)
	except:
		std1=0
	try:
		std2=stdev(packet_lengths)
	except:
		std2=0
	cipher_lst=["AES128-GCM-SHA256","AES128-SHA","AES128-SHA256","AES256-GCM-SHA384","AES256-SHA","AES256-SHA256","ES-CBC3-SHA","DHE-RSA-AES256-GCM-SHA384","DHE-RSA-AES256-SHA","ECDHE-ECDSA-AES128-GCM-SHA256","ECDHE-ECDSA-AES256-GCM-SHA384","ECDHE-RSA-AES128-GCM-SHA256","ECDHE-RSA-AES128-SHA256","ECDHE-RSA-AES256-GCM-SHA384","ECDHE-RSA-AES256-SHA","ECDHE-RSA-AES256-SHA384"]	
	z=0
	i=0
	cipher_arr=[0]*len(cipher_lst)
	for c in cipher_lst:
		i+=1
		##print(c)
		if cipher == c:
			z=1
			cipher_arr[i]=1
			break
	
	#print(cipher)
	#print(cipher_arr)
	arr=[in_bytes,out_bytes,in_packets,out_packets,float(total_time),min(inter_arrival_time),max(inter_arrival_time),mean(inter_arrival_time),std1,min(packet_lengths),max(packet_lengths),mean(packet_lengths),std2]
	#arr.append(cipher_arr)
	for a in cipher_arr:
		arr.append(a)
	#print(arr)
	return arr

# A function to run a ML model on nw flow properties
def predict_NPA(data, model_path):
	#model_path = 'packet_analyzer_model_RF.sav'
	model = pickle.load(open(model_path, 'rb'))
	return model.predict_proba(data)[0]

def getDomain(url):
	if url.find('https://')==0:
		url = url[8:]
	elif url.find('http://')==0:
		url= url[7:]
	
	if url.find('/')!=-1:
		url = url[:url.find('/')]
	return url
	
def removeWWW(url):
	if url.find('https://www.')==0:
		url = 'https://'+url[12:]
	elif url.find('http://www.')==0:
		url = 'https://'+url[11:]
	elif url.find('www.')==0:
		url = url[4:]
	return url


def checkAlive(url):
	resp = requests.get(url, verify = False, timeout = 0.5)
	if resp.status_code==200:
		return True
	return False
	
	

# preparing the data fpr training the model
X = []
Y = []

safe = 0
unsafe = 0

with open('Dataset_Complete_Sorted_unique.txt', 'r') as csvfile:
	csvreader = csv.reader(csvfile, delimiter='\t') 
	for row in csvreader:
		link = row[0]
		label = row[1]
		if label>0:
			label = 1
		pred_featureless, pred_combined = fun_pred_combined('./model/', files, in_names, op_names, removeWWW(link))
		domain = getDomain(link)
		cert_extracted, cert = get_certificate(domain)
		rf_prediction = 0.6
		dt_prediction = 0.6
		knn_prediction = 0.6
		if cert_extracted==1:
			cert = x509.load_der_x509_certificate(cert, default_backend())
			print("===========cerrrrrrrrrrrs================")
			data = extract_properties_cert(cert)
			_, rf_prediction =predict_certificate_prop([data], 'rf_model.sav')
			_, dt_prediction = predict_certificate_prop([data], 'dt_model.sav')
			_, knn_prediction = predict_certificate_prop([data], 'knn_model.sav')
		
		nw_prediction_value = 0.5
		if checkAlive(link):
			data_nw_flow = extract_flow_data(domain)
			if data_nw_flow[-1]=='DNS not found':
				nw_prediction_value = 0.5
			elif data_nw_flow[-1]=='requests error':
				nw_prediction_value = 0.7
			elif data_nw_flow[-1]=='no packets captured':
				nw_prediction_value = 0.6
			else:
				_, nw_prediction_value  = predict_NPA([data_nw_flow], 'rf_model_pa.sav')
		
		with open('outputFile.txt', 'a') as f:
			f.write(link + '\t' + str(pred_featureless) + '\t' + str(pred_combined) + '\t' + str(rf_prediction) + '\t' + str(dt_prediction) + '\t' + str(knn_prediction) + '\t' + str(nw_prediction_value) + 't' + str(label) + '\n')
	success = 1
	
