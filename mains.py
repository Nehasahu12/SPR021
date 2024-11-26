
import nts_function as nts
import pandas as pd
import os
import sys
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# os.mkdir("image_report")
image_report = "image_report"

# Iterate over each file
# file_names = "2493267_24_02_2023_15_21_53.json"

# Replace with input CSV file path
input_file = "signal_result.csv"

# Replace with output CSV file path
output_file = "graphs.csv"
# Specify the column index to be removed
column_index = 9

# Process each file and generate results
# Read From Folder
# file_names = os.listdir("data2/")

# If you want o read as system arguments. 
# it can accept multiple file at the same args
# Read From Argumetns

file_names = sys.argv[1:]
file_names = ["2510394_31_08_2023_16_13_45.json"]
for file_n in file_names:
    if file_n.endswith(".json"):
        # file_name = "data2/2390720_02_06_2023_11_2_56.json"
        
        file_name = f"/home/datasets/test/ECG/{file_n}"

        file_data = nts.process_the_signal(file_name)
        # print(f"Success: {file_data}")

        # Create the report
        pdf_file = f"/home/datasets/test/pdf_result/{file_n.replace('.json', '.pdf')}"
        nts.create_the_report(input_file, output_file, column_index, pdf_file=pdf_file)
        # nts.process_signal_vaildation(input_file)


        heart_rate = int(file_data["HeartRate"])
        signal_to_noise_ratio = float(file_data["SNR"])
        window_flag = True          # read from the file.and
        code = 12345    

        

        RESP = nts.process_signal_vaildation(heart_rate, signal_to_noise_ratio, window_flag, code)

        print(f"ABCD: {RESP}")


# print("Pogram Comleted!")
# sys.exit()