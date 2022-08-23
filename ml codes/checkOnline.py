import requests
import csv
import sys


def check_online(url):
	resp = requests.get(url, verify = False, timeout = 0.5)
	if resp.status_code == 200:
		return True
	return False
	
def check_online_urls_from_file(inFile, outFile):
	# the inFile and outFile are txt files 
    online = 0
    with open(inFile, 'r') as csvfile: 
        csvreader = csv.reader(csvfile, delimiter='\t') 
        for row in csvreader:   
            link = row[0]
            try:
                if check_alive(link):
                    with open(outFile, 'a') as f:
                        f.write(link + '\t' + str(row[1])+'\n')
                        online = online + 1
            except Exception as e:
                Error_flag =1
    print('Found {} urls to be online'.format(online))
	
if __name__=="__main__":
	
	if len(sys.argv) == 2:
		if check_online(sys.argv[1]):
			print("The URL is online")
		else:
			print("The URL is currently offline")
	else:
		check_online_urls_from_file(sys.argv[1], sys.argv[2])