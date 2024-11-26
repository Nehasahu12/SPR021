# app.py
from time import sleep
from flask import Flask 
from flask_autoindex import AutoIndex
import os

# sleep(10)

app = Flask(__name__)

# /home/pi/ECG_GUI/
ppath = "/home/yusuf/Downloads/AllData/" # update your own parent directory here

try:
    sub_folder = os.listdir(ppath)
    for f in sub_folder:
        if "." not in f:
            if len(os.listdir(ppath+"/"+f)) == 0:
                print("There is no folder named")
                os.rmdir(ppath+"/"+f)
            else:
                print("There is a folder named: " + f)
except:
    print("Errror in remove folder")
    pass

app = Flask(__name__)
# app.debug = True
AutoIndex(app, browse_root=ppath)

if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0",port=6000)

