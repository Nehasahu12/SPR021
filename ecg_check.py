import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
import json

qrs_peak=0
def check_error_code(data):
    global ecg_data
    global qrs_peak
    # Check if the 'payLoad' key exists in the data
    # print(data['payLoad']['data'])
    if 'payLoad' not in data:
        res="Missing 'payLoad' in data"
        code=4001
        status="failed"
        result={"status":status,"code":code,"response":res}
        result=json.dumps(result)
        # print(result)
        return result                                                                           
    payLoad = data['payLoad']
    
    
    # Check if the 'data' key exists in the payload
    if 'data' not in payLoad:
        res="Missing 'data' in payload"
        code=4002
        status="failed"
        result={"status":status,"code":code,"response":res}
        result=json.dumps(result)
        # print(result)
        return result
    
    data_values = payLoad['data']  
    
    # print(set(data_values.split(",")))
    # Check if the 'data' value is empty or contains only zeros
    # zeros_count = data_values.count('0')  # counting the character “0” in the given string
    # # print("The count of '0' is", zeros_count)
    # if zeros_count>400:
    # unique_values, counts = np.unique(data_values, return_counts=True)

    counted_values = Counter(data_values.split(","))
    # print(counted_values)
    for value, count in counted_values.items():
        # print(count)
        # if count < 400:
            # ecg_data=ecg_data+count
        qrs_peak=qrs_peak+1
            # # pass
            # print("Value", value, "appears", count, "times in the ECG data." , qrs_peak)
    # print(len(data_values.split(",")))
    if len(data_values.split(",")) < 800:

        res="Insufficient data points"
        code=4004
        status="failed"
        result={"status":status,"code":code,"response":res}
        result=json.dumps(result)
        # print(result)
        return result

    # if qrs_peak < 400:
    #     res="Empty or all-zero 'data' values"
    #     code=4003
    #     status="failed"
    #     result={"status":status,"code":code,"response":res}
    #     result=json.dumps(result)
    #     # print(result)
    #     return result
            
    # Check if the length of 'data' is less than a certain threshold
    
            
  
#     # If none of the error conditions are met, return None to indicate no error
    return None



    
