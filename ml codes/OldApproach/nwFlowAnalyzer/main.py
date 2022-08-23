import os
import csv
import pandas as pd
from code import flow_data
import requests


result_lst=[]
with open('Dataset_Complete_Sorted_unique.txt', 'r') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter='\t')
	i = 0
	success = 0
	for row in csv_reader:
	
		url = row[0]
		
		i+=1
		code = 400
		try:
			resp = requests.get(url, verify = False, timeout = 0.5)
			code = resp.status_code
		except:
			code = 400
		if code!=200:
			continue
			
		if url.find("https://")==0:
			url = url[8:]
		elif url.find("http://")==0:
			url = url[7:]
		else:
			url = url
		if url.find('/')!=-1:
			url = url[:url.find('/')]
		try:
			obj=flow_data(url)
			arr=obj.data(url)
		except:
			continue
		label = row[1]
		success+=1
		print (str(i)+ ' ' +str(success) + ' success' )
		arr.append(label)
		arr.append(url)
		result_lst.append(arr)
		
print ("Done")
cols=['in_bytes','out_bytes','in_packets','out_packets','time','src_port','dst_port','time_min','time_max','time_mean','time_std','len_min','len_max','len_mean','len_std','cipher','eror', 'label', 'url']
df = pd.DataFrame(result_lst, columns=cols)
df.to_csv("data.csv")