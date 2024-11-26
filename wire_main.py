"""
Objective: Arrhythmia classification in Lead II only

"""

# Importing necessary module
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from copy import deepcopy
from CT_PR import lead_II
from CT_PR import qrs_cal
from utils import prograssion as pro
from utils import stj_point
from CT_PR import t_wave_finder_in_all
import warnings
import neurokit2 as nk
from utils.color_segments import ColorQrs
from utils.custom_print import print
from Arr import Arrhythmia_classification
from ecg_check import check_error_code
import sys
import nts_function as nts

def check_signal_health(file_n):
    if file_n.endswith(".json"):
        # file_name = "2390720_02_06_2023_11_2_56.json"                
        file_name = f"{file_n}"
        file_data = nts.process_the_signal(file_name)
        # print(f"Success: {file_data}")        
        heart_rate = int(file_data["HeartRate"])
        signal_to_noise_ratio = float(file_data["SNR"])
        window_flag = True 
        # read from the file.and        
        code = 12345
        RESP = nts.process_signal_vaildation(heart_rate, signal_to_noise_ratio, window_flag, code)
        return RESP
    else:
        return False

try:
    warnings.filterwarnings('ignore')
    # configure matplotlib figure size to 15x4 inch
    plt.rcParams["figure.figsize"] = (12, 3)
    filePath=sys.argv[1]

    # filePath=sys.argv[1]
    if filePath == "1":
        res="API key missing"
        code=403
        status="failed"
        result={"status":status,"code":code,"response":res}
        result=json.dumps(result)
        print(result)
        sys.exit()
    elif filePath == "2":
        res="API key invalid"
        code=405
        status="failed"
        result={"status":status,"code":code,"response":res}
        result=json.dumps(result)
        print(result)
        sys.exit()
    # print(filePath)
    else:
        with open(filePath,'r') as f:
            data = eval(f.read())
            resp = check_signal_health(filePath)
            resp=json.loads(resp)
            a=resp["status"]
            # print(resp)
            # a=check_error_code(data)
            
            # file_data = nts.process_the_signal(filePath)
            # pdf_file = f"../pdf_result/{file_n.replace('.json', '.pdf')}"
            # nts.create_the_report(input_file, output_file, column_index, pdf_file=pdf_file)
            # heart_rate = int(file_data["HeartRate"])
            # signal_to_noise_ratio = float(file_data["SNR"])
            # window_flag = True          # read from the file.and
            # code = 12345  
            # a = nts.process_signal_vaildation(heart_rate, signal_to_noise_ratio, window_flag, code)
            # print(f"ABCD: {a}")

            if a != 'NA':
                print(resp)
                # print(" error detected.") 
            else:        
                #print("No error detected.")                        
                
                lead_data = data['payLoad']['data']
                amp_gain = data['leadGain']
                lead_data =lead_data.split(',')
                
                lead_data = data['payLoad']['data']
                lead_data =lead_data.split(',')
                lead_data =[float(i) for i in lead_data]
            # Convert each ADC value to millivolts
                reference_voltage_mv = 3300 # in mV (3.3V)
                resolution_bits = 12  # 12-bit ADC
                max_adc_value = (2 ** resolution_bits) - 1

                # Applying the conversion
                lead_data = [(value / max_adc_value) * reference_voltage_mv for value in lead_data]
          
                lead_data =[float(i) for i in lead_data]
                headerlist = ['Lead_II']
                csv_file = './output.csv'
                df = pd.DataFrame(lead_data)
                #df=df/100
                df.to_csv(csv_file, header=headerlist, index=False)
                # plt.plot(df)
                # plt.show()
                df = pd.read_csv(csv_file)
                lead_i_data = df["Lead_II"]     
                
                signal_gain = 1000
                sampling_rate =76 # round((len(df["Lead_II"])+400) / 13)
                per_sample = 1 / sampling_rate
                pid = 12345

                # Creating a QRS Object
                QRS = qrs_cal.CalculateQRS(df=df, pid_number=pid, sampling_rate=sampling_rate, filter_mode=3,
                                                amp_gain=amp_gain, low_cutoff_freq=40, high_cutoff_freq=0.56)
                    # print("Line 127")
                QRS.clean_all_data()

                    # QRS.check_m_shape()
                QRS.scatter_r_peak()
                QRS.fix_qrs_peak()
                QRS.scatter_qrs(ind_v=True, show=False) # If don't want to see graph make 'show=False'

                qrs_interval = QRS.get_qrs()
                QRS.get_heart_rate(sampling_rate=sampling_rate)
                rr_interval = QRS.get_rr_interval()

                r_peak_index_val = QRS.get_r_peaks(show=False)  # R, p->peak, i->index and v->value
                leads_name = ["Lead_II"]

                t_time = t_wave_finder_in_all.TDetector(QRS.cleaned_signal, r_peak_index_val, sampling_rate=sampling_rate,
                                                            margin=0, baseline=QRS.isoelectric)
                t_time.adjust_baseline(show=False)
                t_time.split_t_segments(show=False)
                t_time.find_t_peak(show=False)
                t_time.split_qrs_segments(show=False)
                jp = t_time.get_j_point_from_2d(show=False)

                result = {"perlead": []}

                led = "Lead_II"
                plt.plot(QRS.cleaned_signal[led])
                I = lead_II.LeadII(QRS.cleaned_signal[led], sampling_rate=sampling_rate,amp_gain=amp_gain)
                # I = lead_II.LeadII(QRS.cleaned_signal[led], sampling_rate=sampling_rate)
                I.set_r_peaks(r_pears=r_peak_index_val[led], baseline=QRS.isoelectric[led], adjust_sig=False, show=False)
                I.calculate_p_peaks(show=False, r=False) # If don't want to see graph make 'show=False' r also
                I.get_atrial_rate()
                I.get_pr_interval(QRS.qrs_segments[led]["index"], per_sample=per_sample)
                I.get_basic_t(show=False, j_point=jp[0]) # If don't want to see graph make 'show=False'
                # I.get_qt_interval(per_sample=per_sample, rr_interval=QRS.get_rr_interval(led))
                # I.get_qt_class()
                # I.get_pr_type()
                # I.get_qrs_area()
                I.rate_classification()
                result["perlead"].append(I.get_report())
                # print(result)


                # Arrhythmia classification below
                leads = "Lead_II"
                res = result["perlead"][0]
                analy = Arrhythmia_classification(QRS.cleaned_signal[led], r_peak_index_val)
                arth = analy.classify_data(res, leads).strip()
                result["perlead"][0][leads]["Diagonosis"] = arth
                result=json.dumps(result)
                # result=result.replace("'","\"")
                print(result)
except Exception as e:
    result={"status":"Error","code":500,"response":str(e)}
    print(result)
            
                
