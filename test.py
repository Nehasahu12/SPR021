import subprocess
import requests
import json
import time

print('reading current time')
time.sleep(1)
timeUpdated=False

try:
    print('reading time from server')
    url='http://172.16.5.250:1880/GetCurrentDateTime'
    r=requests.get(url)
    print(r.text)
except Exception as e:
    print("Error: " + str(e))