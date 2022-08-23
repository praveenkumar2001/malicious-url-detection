import sys
import csv
import ssl
import socket
import OpenSSL
import datetime

from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography import utils
from cryptography.x509.oid import NameOID, ObjectIdentifier, ExtensionOID
from cryptography.hazmat.primitives import hashes

def get_certificate(url):
	hostname = url
	port = 443
	try:
		context = ssl.create_default_context()
		conn = ssl.create_connection((hostname, port), timeout = 0.5)
		sock = context.wrap_socket(conn, server_hostname = hostname)
		derCert = sock.getpeercert(True)
		return 1, derCert
	except Exception as e1:
		try:
			pemCert = ssl.get_server_certificate((hostname, port))
			derCert = ssl.PEM_cert_to_DER_cert(pemCert)
			return 1, derCert
		except Exception as e2:
			return 0,0
		return 0,0
	return 0,0

def extract_properties(derCert):
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

def getDomain(link):
	if link.find('https://')==0:
		link = link[8:]
	if link.find('http://')==0:
		link = link[7:]
	if link.find('/')!=-1:
		link = link[:link.find('/')]
	if link.find('%2F')!=-1:
		link = link[:link.find('%2F')]
	if link.find('%2f')!=-1:
		link = link[:link.find('%2f')]
	return link

if __name__=="__main__":

	if len(sys.argv)==2:
		link = sys.argv[1]
		cert_extracted, cert = get_certificate(link)
		if cert_extracted==1:
			properties = extract_properties(cert)
			print(len(properties))
			print(properties)
		else:
			print("cert not extracted")
	else:
		inFile = sys.argv[1]
		outFile = sys.argv[2]
		sucess = 0
		fail = 0
		i = 0
		with open(inFile, 'r') as csvfile: 
			csvreader = csv.reader(csvfile, delimiter='\t') 
			for row in csvreader:   
				link = row[0]
				link = getDomain(link)
				label = int(row[1])
				i+=1
				if label>0:
					label = 1
				cert_extracted, cert = get_certificate(link)
				
				if cert_extracted==1:
					
					properties = extract_properties(cert)
					with open(outFile, 'a') as f:
						f.write(link + '\t')
						for p in properties:
							f.write(str(p) + '\t')
						f.write(str(label) + '\n')
					sucess+=1
					print(str(i) + '  ' + str(sucess) + ' cert_extracted')
				else:
					fail+=1
		print("Sucess : ", sucess)
		print("Failed : ", fail)
	print("End")
	
