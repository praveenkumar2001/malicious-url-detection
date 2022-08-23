# Code to read the URLs from a file
# and then check that URL using the 
# API provided by Virus Total.
#
# This file also contains the 6 API 
# keys used for communicating with 
# VirusTotal. They allow 4 query/min.
#
# The data is also written to op file.

import requests
import time
import csv

def getReponseVirusTotal(url, apiKey):
	params = {'apikey': apiKey, 'resource':url,'scan':'1'}
	response = requests.post('https://www.virustotal.com/vtapi/v2/url/report',params=params, verify=False)
	json_response = response.json()
	return [json_response['positives'], json_response['total']]


#===================Keys====================#	
Dhanrmendra = '01b8ce94dec75c2c2155ee2399b2ee2f16450a719b76067369bd0eebc9fcfb1d'
Tanmay = '42b9c7d8bd486897873d132ca65da8e4bb6ce497f3529082db5c6737e0b3cd70'
Amit = '4e4ccc54b273e5e16ec832a19aa2ee6d1dac9dfab8782705976a603e8419ea9a'
Aditya = 'e64a469e4783686c4db154d158aa10df7728e72c749c942092775a435ef64e6a'
Rohan = '80574a46ff03bdbcb69e119b867bc3a0445853ea0eac4d96e6f2decce68082f2'
Dewesh = '88f6954061b8f649015ea1ffa788f81cbf9e35838cbb49b5d543929439c10f2f'


keys = [Dhanrmendra, Tanmay, Amit, Aditya, Rohan, Dewesh]

#==========input & output file names=========#
inpuFile_name = "Dataset_Complete_Sorted_unique.txt"
outputFile_name = "Dataset_Complete_Sorted_Verified_Unique.txt"



with open(inpuFile_name, 'r') as csvfile:
	csvreader = csv.reader(csvfile, delimiter='\t')
	for row in csvreader:
		url = row[0]			
		try:
			answer = getReponseVirusTotal(url, keys[i%6])
			print(answer[0], answer[1])
			
			try:
				with open(outputFile_name, 'a') as f:
					f.write(url + '\t' + str(answer[0]) + '\t' + str(answer[1]) + '\t' + '\n')
			except:
				error = 1
			
		except Exception as e:
			print("\n===============error=================")
			print(i, e)
		
		if i%6==0:
			time.sleep(16)
			
print("\n============Success================")