#takes the URL as command line argument

#reading from the command line
import sys
import pickle
#establising connection and downloading the certificate
import ssl
import socket
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
import pandas as pd
import numpy as np
import csv
import sys 
import socket as socket
import ssl as ssl
from statistics import mean,stdev

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


def extract_properties_cert(derCert):
	cert = x509.load_der_x509_certificate(derCert, default_backend())
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
	

	
def predict_certificate_prop(data, model_path):
	#model_path = 'certificate_classifier_model.sav'
	model = pickle.load(open(model_path, 'rb'))
	return model.predict_proba(data)	

	
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

	print(ip_arr[0])
	if len(ip_arr)==0:
		#print('DNS not found')
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
    
        
	print(in_packets)
	print(out_packets)
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
		#print(c)
		if cipher == c:
			z=1
			cipher_arr[i]=1
			break
	
	print(cipher)
	print(cipher_arr)
	arr=[in_bytes,out_bytes,in_packets,out_packets,float(total_time),min(inter_arrival_time),max(inter_arrival_time),mean(inter_arrival_time),std1,min(packet_lengths),max(packet_lengths),mean(packet_lengths),std2]
	#arr.append(cipher_arr)
	for a in cipher_arr:
		arr.append(a)
	print(arr)
	return arr


def predict_NPA(data, model_path):
	#model_path = 'packet_analyzer_model_RF.sav'
	model = pickle.load(open(model_path, 'rb'))
	return model.predict_proba(data)

url = sys.argv[1]
extracted, cert = get_certificate(url)

if extracted==1:
	data = extract_properties_cert(cert)
	rf_prediction = predict_certificate_prop([data], 'rf_model.sav')
	dt_prediction = predict_certificate_prop([data], 'dt_model.sav')
	knn_prediction = predict_certificate_prop([data], 'knn_model.sav')
	
	rf_prediction_value, _ = rf_prediction[0]
	dt_prediction_value, _ = dt_prediction[0]
	knn_prediction_value, _ = knn_prediction[0]
	
	
	
else:
	print('Not extracted')
	rf_prediction_value = dt_prediction_value = knn_prediction_value = 0.5
	print('[[0.5 0.5]]')

print(rf_prediction_value, dt_prediction_value, knn_prediction_value)
data_nw_flow = extract_flow_data(url)
if data_nw_flow[-1]=='DNS not found':
	print('DNS not found')
	nw_prediction_value = 0.5
	print('[[0.5 0.5]]')
elif data_nw_flow[-1]=='requests error':
	print('requests error')
	print('[[0.2 0.8]]')
	nw_prediction_value = 0.2
elif data_nw_flow[-1]=='no packets captured':
	print('no packets captured')
	print('[[0.4 0.6]]')
	nw_prediction_value = 0.4
else:
	print("\n=============== ", len(data_nw_flow))
	nw_prediction  = predict_NPA([data_nw_flow], 'rf_model_pa.sav')
	nw_prediction_value , _ = nw_prediction[0]
	

safety_level = (rf_prediction_value + dt_prediction_value*0.5 + knn_prediction_value + nw_prediction_value*2)/4.5

print('safe : ', safety_level)
