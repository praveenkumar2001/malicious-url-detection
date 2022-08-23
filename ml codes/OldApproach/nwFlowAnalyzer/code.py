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

class flow_data:
	def __init__(self,target):
		url=target
		#print "##########"
		#print (url)
	def data(self,url):
		#print sys.argv[1]
		#url=sys.argv[1]
		process1 = sub.Popen(["nslookup", url], stdout=sub.PIPE)
		#print(process1.communicate()[0])
		output = process1.communicate()[0].decode('utf-8').split('\n')
		
		error='no error'
		ip_arr = []
		zeros=[0]*15
		zeros.append('0')
		for data in output:
			if 'Address' in data:
				ip_arr.append(data.replace('Address: ',''))	
		ip_arr.pop(0)

		#print (ip_arr)
		if len(ip_arr)==0:
			zeros.append('DNS not found')
			return zeros

		s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		ssl_sock = ssl.wrap_socket(s,cert_reqs=ssl.CERT_REQUIRED,ca_certs='/etc/ssl/certs/ca-certificates.crt')
		try:
			ssl_sock.connect((ip_arr[0], 443))
			#print repr(ssl_sock.getpeername())
			#print (ssl_sock.cipher())
			cipher=ssl_sock.cipher()[0]
		except Exception as e:
			error=e
			cipher=0
		
		#cmd='tcpdump -w 001.pcap -i eth0 tcp and src 172.217.27.196 or dst 172.217.27.196'
		process2 = sub.Popen(('tcpdump', '-w', '001.pcap', '-i', 'eth0','-v', 'tcp', 'and', 'src', ip_arr[0], 'or','dst',ip_arr[0]), stdout=sub.PIPE)

		url='http://'+ url
		#r=wget.download(url)
		#r = requests.get(url)
		try:
			r = requests.get(url, timeout=10)
		except Exception as e:
			zeros.append(e)
			process2.terminate()
			return zeros
		time.sleep(2)
		process2.terminate()
		#os.killpg(os.getpgid(p.pid), signal.SIGTERM)  # Send the signal to all the process groups


		cmd='tshark -r 001.pcap -T fields -e _ws.col.Time -e _ws.col.Source -e _ws.col.Destination -e _ws.col.SrcPort -e _ws.col.DstPort -e  _ws.col.Protocol -e _ws.col.Length -E separator=, -E occurrence=f > pcap_traffic.csv'
		#print cmd
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
		with open('pcap_traffic.csv', 'r') as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=',')
	
			for row in csv_reader:
				if int(row[6])==0:
					break
				if z==0 and row[1]==ip_arr[0]:
					src_port=int(8001)
					dst_port=int(8001)
					z=1
				if i>0:
					inter_arrival_time.append(float(row[0])-temp)
					temp=float(row[0])
				packet_lengths.append(int(row[6]))
				#print row[0]
				if row[6]=='0':
					break
				i+=1
				total_time=row[0]
				if row[1]==ip:
					#print i
					out_bytes+=int(row[6])
					out_packets+=1
				else:
					#print i
					in_bytes+=int(row[6])
					in_packets+=1
		if i==0:
			zeros.append('no packets captured')
			return zeros

		
		#print (in_packets)
		#print (out_packets)
		try:
			std1=stdev(inter_arrival_time)
		except:
			std1=0
		try:
			std2=stdev(packet_lengths)
		except:
			std2=0

		arr=[in_bytes,out_bytes,in_packets,out_packets,total_time,src_port,dst_port,min(inter_arrival_time),max(inter_arrival_time),mean(inter_arrival_time),std1,min(packet_lengths),max(packet_lengths),mean(packet_lengths),std2,cipher,error]
		#print arr
		return arr

