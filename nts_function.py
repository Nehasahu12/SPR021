# import os
# import json
# import csv
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import signal
# import pandas as pd
# from matplotlib.backends.backend_pdf import PdfPages
# import matplotlib.font_manager as fm
# from copy import deepcopy

# # Constants
# flc = 25
# fhc = 0.1
# N1 = 0
# N2 = 1000
# fs = 100
# image_report = "/home/datasets/test/image_report"

# # # Butterworth filter parameters
# # w_high = fhc / (fs / 2)
# # w_low = flc / (fs / 2)
# # b_high, a_high = signal.butter(4, w_high, 'high')
# # b_low, a_low = signal.butter(4, w_low, 'low')

# #apply high pass filter
# w = fhc/(fs/2) # normalized frequency
# b,a= signal.butter(4,w,'high')

# w = flc/(fs/2) # normalized frequency
# b,a= signal.butter(4,w,'low')

# sampling_rate = fs

# def detect_r_peaks(filteredArray, fs):
#     # Differentiate the signal
#     diff_ecg = np.diff(filteredArray)

#     # Square the differentiated signal
#     squared_ecg = np.square(diff_ecg)

#     # Apply a moving average filter
#     window_length = int(0.08 * fs)
#     moving_average = np.convolve(squared_ecg, np.ones(
#         window_length)/window_length, mode='same')

#     # Find the R-peaks using the maxima detection
#     r_peaks, _ = signal.find_peaks(
#         moving_average, height=0.3*np.max(moving_average), distance=int(0.3*fs))

#     return r_peaks


# def get_snr(lead_II):
#     # Applying 0.5 Hz Highpass filter to signal
#     lead_II_after_high_pass = lead_II

#     # Compute the frequency spectrum using FFT
#     frequency_spectrum = np.asarray(np.fft.fft(lead_II_after_high_pass), float)
#     frequencies = np.abs(np.asarray(np.fft.fftfreq(
#         len(lead_II_after_high_pass), 1 / sampling_rate), float))

#     start = int(len(frequency_spectrum)/2)
#     frequency_spectrum = frequency_spectrum[0:start]
#     frequencies = frequencies[0:start]

#     # Define the frequency range
#     lower_bound = 2
#     upper_bound = 40

#     # Filter frequencies within the specified range
#     ecg_signal_indices = np.where(
#         (frequencies > lower_bound) & (frequencies < upper_bound))
#     ecg_signal_spectrum = np.asarray(
#         np.array([frequency_spectrum[i] for i in ecg_signal_indices]), float)[0]

#     # Define the frequency range
#     noise_lower_bound = 2
#     noise_upper_bound = 40

#     # Filter frequencies within the specified range
#     noise_signal_indices = np.where(
#         (frequencies <= noise_lower_bound) | (frequencies >= noise_upper_bound))
#     noise_signal_spectrum = np.asarray(
#         np.array([frequency_spectrum[i] for i in noise_signal_indices]), float)[0]

#     # Compute the power spectrum
#     ecg_signal_power_spectrum = np.mean(np.square(ecg_signal_spectrum))
#     noise_signal_power_spectrum = np.mean(np.square(noise_signal_spectrum))

#     signal_to_noise_ratio = 10 * np.log10(ecg_signal_power_spectrum / noise_signal_power_spectrum)
#     return signal_to_noise_ratio


# def process_signal_vaildation(heart_rate=0, signal_to_noise_ratio=0,
#                 window_flag=False, code="", final_code=3000):
#     if 24 > heart_rate or heart_rate > 600:
#         res = "Signal failed"
#         code = 3001
#         status = "failed"
#         result = {"status": status, "code": code, "response": res, "final_code":final_code}
#         result = json.dumps(result)
#         return result
#     elif signal_to_noise_ratio < 0.5:
#         res = "Signal failed"
#         code = 3002
#         status = "failed"
#         result = {"status": status, "code": code, "response": res, "final_code":final_code}
#         result = json.dumps(result)
#         return result

#     elif window_flag == False:
#         res = "Signal failed"
#         code = 3003
#         status = "failed"
#         result = {"status": status, "code": code, "response": res, "final_code":final_code}
#         result = json.dumps(result)
#         return result

#     elif code == 3001 or code == 3002 or code == 3003:
#         res = "Signal failed"
#         code = 3000
#         status = "failed"
#         result = {"status": status, "code": code, "response": res, "final_code":final_code}
#         result = json.dumps(result)
#         return result

#     else:
#         result = {"status": "NA", "code": "NA", "response": "NA", "final_code": "NA"}
#         result = json.dumps(result)
#         return result

# # new function


# def process_the_signal(file_name, fs=100):
#     result = {"file_name": "", "SNR": 0, "Status": "", "pantompkin": "",
#                "HeartRate": 0, "window": "-", "image": "", "signal status": "", "message": "NA"}

#     with open(file_name) as json_file:
#         try:
#             data = json.load(json_file)
#             # print(data)
#             dicData = data
#             # temp = {"file_name": file_name, "SNR": 0, "Status": "", "pantompkin": "", "HeartRate": 0,
#             #         "window": "-", "image": "", "signal status": "", "message": "NA"}
#             if 'payLoad' not in dicData:
#                 print("Error ==>> There is No payload")
#                 result["message"] = "Payload Not Found"
#             else:
#                 plt.figure()
#                 plt.title(file_name)

#                 result["file_name"] = file_name
#                 leadData = [float(i) for i in dicData["payLoad"]["data"].split(",")]
#                 leadArray = (np.array(leadData)/100)
#                 plt.plot(leadArray)
#                 plt.savefig(
#                     f"{image_report}/{os.path.basename(file_name).split('.')[0]}.png")
#                 result["image"] = f"{image_report}/{os.path.basename(file_name).split('.')[0]}.png"
#                 N = len(leadArray)
                
#                 filteredArray = signal.filtfilt(b, a, leadArray)
#                 ecg_lead_signal = deepcopy(filteredArray)
#                 r_peaks = detect_r_peaks(filteredArray, fs)
#                 # print(f"R peak: {r_peaks}")
#                 if len(r_peaks) > 1:
#                     heart_rate = np.round(60 / (np.mean(np.diff(r_peaks)) / fs))
#                 else:
#                     heart_rate = 0
#                 # print(f"Heart Rate: {heart_rate}")
#                 result["HeartRate"] = heart_rate
#                 if 24 <= heart_rate <= 600:
#                     # print("Pan Tompkins: Signal passed")
#                     result["pantompkin"] = "Passed"
#                 else:
#                     # print("Pan Tompkins: Signal not passed")
#                     result["pantompkin"] = "Not Passed"

#                 signal_to_noise_ratio = get_snr(ecg_lead_signal)
#                 if signal_to_noise_ratio > 0.5:
#                     # print(f"SNR: {signal_to_noise_ratio} dB\tSignal passed")
#                     result["SNR"] = signal_to_noise_ratio
#                     result["Status"] = "passed"
#                 else:
#                     # print(f"SNR: {signal_to_noise_ratio} dB\tSignal not passed")
#                     result["Status"] = "not passed"
#                     result["SNR"] = signal_to_noise_ratio
#                 mean = np.mean(filteredArray)
#                 variance = np.var(filteredArray)

#                 # Set thresholds for mean and variance differences
#                 threshold_mean = 0.05 * mean
#                 threshold_variance = 0.05 * variance

#                 # Iterate through the signal in segments or windows
#                 window_size = 200  # Window size in samples
#                 stride = 20  # Stride between windows in samples

#                 window_flag = True
#                 for i in range(0, len(filteredArray) - window_size, stride):
#                     # Extract the current window of the signal
#                     window = filteredArray[i:i+window_size]

#                     # Calculate the mean and variance of the current window
#                     window_mean = np.mean(window)
#                     window_variance = np.var(window)
#                     if abs(window_mean - mean) > threshold_mean or abs(window_variance - variance) > threshold_variance:
#                         if mean < 1:
#                             window_flag = False
#                 if window_flag:
#                     result["window"] = "signal passed"
#                 else:
#                     result["window"] = "signal not passed"

#             if result["pantompkin"] == "Passed" and result["Status"] == "passed" and result["window"] == "signal passed":
#                 signal_status = "Signal is passed"
#             else:
#                 signal_status = "Signal not passed"
#                 result["signal status"] = signal_status

#                 # print("============================")
#                 # result.append(temp)

#         except json.JSONDecodeError as e:
#             print(f"***** Error loading JSON from {file_name}: {e}")
#             return result
#         except Exception as e:
#             print(f"***** There is Error in the generate_report: {e}")
#             return result

#     df = pd.DataFrame([result, result])
#     df.to_csv("signal_result.csv")
#     # print(f"Result: {result}")
#     # print("Data Frame/n", df)
#     return result


# def create_the_report(input_file, output_file, column_index=0, pdf_file="test.pdf"):
#     # image_folder = "image_report"
#     csv_file = "graph_output.csv"

#     input_file = "graphs.csv"
#     output_file = "graph_output.csv"

#     # Read the input CSV file and exclude the desired column
#     with open("signal_result.csv", "r") as file:
#         reader = csv.reader(file)
#         data = [row for row in reader]

#     modified_data = []
#     for row in data:
#         modified_row = row[:column_index] + row[column_index+1:]
#         modified_data.append(modified_row)

#     # Write the modified data to the output CSV file
#     with open("graphs.csv", "w", newline="") as file:
#         writer = csv.writer(file)
#         writer.writerows(modified_data)

#     row_index = 1  # index of the row to be removed

#     # Read the input CSV file and exclude the desired row
#     with open(input_file, "r") as file:
#         reader = csv.reader(file)
#         data = [row for row in reader if reader.line_num != row_index + 1]

#     # Write the modified data to the output CSV file
#     with open(output_file, "w", newline="") as file:
#         writer = csv.writer(file)
#         writer.writerows(data)

#     # Get a suitable font for increasing the font size
#     font_prop = fm.FontProperties(size=14)

#     # Read the CSV file into a DataFrame
#     data = pd.read_csv(csv_file)

#     # Create a PDF file for saving the merged output
#     pdf_pages = PdfPages(pdf_file)

#     # Iterate through the rows of the CSV file
#     for _, row in data.iterrows():
#         file_name = row['file_name']
#         snr = row['SNR']
#         status = row['Status']
#         pantompkin = row['pantompkin']
#         # message = row['message']
#         heart_rate = row['HeartRate']
#         window = row['window']
#         image_path = row['image']
#         Signal_status = row['signal status']

#         # Check if the image file exists
#         if os.path.isfile(image_path):
#             # Read and process the image file
#             image = plt.imread(image_path)

#             # Create a new page in the PDF file
#             fig = plt.figure(figsize=(8, 11))
#             plt.axis('off')  # Remove x-axis and y-axis
#             plt.imshow(image)

#             # Create a table with the data
#             table_data = [
#                 ['File Name:', file_name],
#                 ['SNR:', snr],
#                 ['Status:', status],
#                 ['Pantompkin:', pantompkin],
#                 # ['Message:', message],
#                 ['Heart Rate:', heart_rate],
#                 ['Window:', window],
#                 ['Signal Status:', Signal_status]
#             ]

#             # Add the CSV data as a table below the plot
#             table = plt.table(cellText=table_data, loc='bottom',
#                               cellLoc='left', colWidths=[0.3, 0.7])
#             table.auto_set_font_size(False)
#             table.set_fontsize(10)
#             table.scale(1, 1.5)
#             # Adjust the positioning of the graph and table
#             # Adjust the rect values as needed
#             fig.tight_layout(rect=[0, 0.25, 1, 1])

#             # Set the properties of table cells to wrap the text
#             for cell in table.get_celld().values():
#                 cell.set_text_props(multialignment='left', wrap=True)
#                 cell.set_height(0.1)  # Adjust the height value as needed

#             # Add the plot to the PDF page
#             pdf_pages.savefig()
#             plt.close()

#         else:
#             print(f"Image file '{image_path}' not found. Skipping...")

#     # Save and close the PDF file
#     pdf_pages.close()


# if __name__ == "__main__":
#     # Replace with input CSV file path
#     input_file = "signal_result.csv"
#     # Replace with output CSV file path
#     output_file = "graphs.csv"
#     # Specify the column index to be removed
#     column_index = 10

#     # Process each file and generate results
#     file_name = "data2/2390720_02_06_2023_11_2_56.json"
#     process_data = process_the_signal(file_name)
#     print(process_data)

#     # # Create the report
#     # create_the_report(input_file, output_file, column_index)


import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.font_manager as fm
from copy import deepcopy

# Constants
flc = 25
fhc = 0.1
N1 = 0
N2 = 1000
fs = 100
image_report = "/home/datasets/test/image_report"


# # Butterworth filter parameters
# w_high = fhc / (fs / 2)
# w_low = flc / (fs / 2)
# b_high, a_high = signal.butter(4, w_high, 'high')
# b_low, a_low = signal.butter(4, w_low, 'low')

#apply high pass filter
w = fhc/(fs/2) # normalized frequency
b,a= signal.butter(4,w,'high')

w = flc/(fs/2) # normalized frequency
b,a= signal.butter(4,w,'low')

sampling_rate = fs

def detect_r_peaks(filteredArray, fs):
    # Differentiate the signal
    diff_ecg = np.diff(filteredArray)

    # Square the differentiated signal
    squared_ecg = np.square(diff_ecg)

    # Apply a moving average filter
    window_length = int(0.08 * fs)
    moving_average = np.convolve(squared_ecg, np.ones(
        window_length)/window_length, mode='same')

    # Find the R-peaks using the maxima detection
    r_peaks, _ = signal.find_peaks(
        moving_average, height=0.3*np.max(moving_average), distance=int(0.3*fs))

    return r_peaks


def get_snr(lead_II):
    # Applying 0.5 Hz Highpass filter to signal
    lead_II_after_high_pass = lead_II

    # Compute the frequency spectrum using FFT
    frequency_spectrum = np.asarray(np.fft.fft(lead_II_after_high_pass), float)
    frequencies = np.abs(np.asarray(np.fft.fftfreq(
        len(lead_II_after_high_pass), 1 / sampling_rate), float))

    start = int(len(frequency_spectrum)/2)
    frequency_spectrum = frequency_spectrum[0:start]
    frequencies = frequencies[0:start]

    # Define the frequency range
    lower_bound = 4
    upper_bound = 40

    # Filter frequencies within the specified range
    ecg_signal_indices = np.where(
        (frequencies > lower_bound) & (frequencies < upper_bound))
    ecg_signal_spectrum = np.asarray(
        np.array([frequency_spectrum[i] for i in ecg_signal_indices]), float)[0]

    # Define the frequency range
    noise_lower_bound = 4
    noise_upper_bound = 40

    # Filter frequencies within the specified range
    noise_signal_indices = np.where(
        (frequencies <= noise_lower_bound) | (frequencies >= noise_upper_bound))
    noise_signal_spectrum = np.asarray(
        np.array([frequency_spectrum[i] for i in noise_signal_indices]), float)[0]

    # Compute the power spectrum
    ecg_signal_power_spectrum = np.mean(np.square(ecg_signal_spectrum))
    noise_signal_power_spectrum = np.mean(np.square(noise_signal_spectrum))

    signal_to_noise_ratio = 10 * np.log10(ecg_signal_power_spectrum / noise_signal_power_spectrum)
    # print (signal_to_noise_ratio)
    return signal_to_noise_ratio


def process_signal_vaildation(heart_rate=0, signal_to_noise_ratio=0,
                window_flag=False, code="", final_code=3000):
    if 24 > heart_rate or heart_rate > 600:
        res = "Signal failed"
        code = 3001
        status = "failed"
        result = {"status": status, "code": code, "response": res, "final_code":final_code}
        result = json.dumps(result)
        return result
    elif signal_to_noise_ratio < 0.5:
        res = "Signal failed"
        code = 3002
        status = "failed"
        result = {"status": status, "code": code, "response": res, "final_code":final_code}
        result = json.dumps(result)
        return result

    elif window_flag == False:
        res = "Signal failed"
        code = 3003
        status = "failed"
        result = {"status": status, "code": code, "response": res, "final_code":final_code}
        result = json.dumps(result)
        return result

    elif code == 3001 or code == 3002 or code == 3003:
        res = "Signal failed"
        code = 3000
        status = "failed"
        result = {"status": status, "code": code, "response": res, "final_code":final_code}
        result = json.dumps(result)
        return result

    else:
        result = {"status": "NA", "code": "NA", "response": "NA", "final_code": "NA"}
        result = json.dumps(result)
        return result

# new function


def process_the_signal(file_name, fs=100):
    result = {"file_name": "", "SNR": 0, "Status": "", "pantompkin": "",
               "HeartRate": 0, "window": "-", "image": "", "signal status": "", "message": "NA"}

    with open(file_name) as json_file:
        try:
            data = json.load(json_file)
            dicData = data
            # temp = {"file_name": file_name, "SNR": 0, "Status": "", "pantompkin": "", "HeartRate": 0,
            #         "window": "-", "image": "", "signal status": "", "message": "NA"}
            if 'payLoad' not in dicData:
                print("Error ==>> There is No payload")
                result["message"] = "Payload Not Found"
            else:
                plt.figure()
                plt.title(file_name)

                result["file_name"] = file_name
                leadData = [float(i) for i in dicData["payLoad"]["data"].split(",")]
                leadArray = (np.array(leadData)/100)
                mean_val = np.mean(leadArray)
                std_dev = np.std(leadArray)
                leadArray = [(x - mean_val) / std_dev for x in leadArray]
                
                plt.plot(leadArray)
                plt.savefig(
                    f"{image_report}/{os.path.basename(file_name).split('.')[0]}.png")
                result["image"] = f"{image_report}/{os.path.basename(file_name).split('.')[0]}.png"
                N = len(leadArray)
                
                filteredArray = signal.filtfilt(b, a, leadArray)
                ecg_lead_signal = deepcopy(filteredArray)
                r_peaks = detect_r_peaks(filteredArray, fs)
                # print(f"R peak: {r_peaks}")
                if len(r_peaks) > 1:
                    heart_rate = np.round(60 / (np.mean(np.diff(r_peaks)) / fs))
                else:
                    heart_rate = 0
                # print(f"Heart Rate: {heart_rate}")
                result["HeartRate"] = heart_rate
                if 24 <= heart_rate <= 600:
                    # print("Pan Tompkins: Signal passed")
                    result["pantompkin"] = "Passed"
                else:
                    # print("Pan Tompkins: Signal not passed")
                    result["pantompkin"] = "Not Passed"

                signal_to_noise_ratio = get_snr(ecg_lead_signal)
                if signal_to_noise_ratio > 0.5:
                    # print(f"SNR: {signal_to_noise_ratio} dB\tSignal passed")
                    result["SNR"] = signal_to_noise_ratio
                    result["Status"] = "passed"
                else:
                    # print(f"SNR: {signal_to_noise_ratio} dB\tSignal not passed")
                    result["Status"] = "not passed"
                    result["SNR"] = signal_to_noise_ratio
                mean = np.mean(filteredArray)
                variance = np.var(filteredArray)

                # Set thresholds for mean and variance differences
                threshold_mean = 0.05 * mean
                threshold_variance = 0.05 * variance

                # Iterate through the signal in segments or windows
                window_size = 200  # Window size in samples
                stride = 20  # Stride between windows in samples

                window_flag = True
                for i in range(0, len(filteredArray) - window_size, stride):
                    # Extract the current window of the signal
                    window = filteredArray[i:i+window_size]

                    # Calculate the mean and variance of the current window
                    window_mean = np.mean(window)
                    window_variance = np.var(window)
                    if abs(window_mean - mean) > threshold_mean or abs(window_variance - variance) > threshold_variance:
                        if mean < 1:
                            window_flag = False
                if window_flag:
                    result["window"] = "signal passed"
                else:
                    result["window"] = "signal not passed"

            if result["pantompkin"] == "Passed" and result["Status"] == "passed" and result["window"] == "signal passed":
                signal_status = "Signal is passed"
            else:
                signal_status = "Signal not passed"
                result["signal status"] = signal_status

                # print("============================")
                # result.append(temp)

        except json.JSONDecodeError as e:
            print(f"***** Error loading JSON from {file_name}: {e}")
            return result
        except Exception as e:
            print(f"***** There is Error in the generate_report: {e}")
            return result

    df = pd.DataFrame([result, result])
    df.to_csv("signal_result.csv")
    # print(f"Result: {result}")
    # print("Data Frame/n", df)
    signal_validation_result = process_signal_vaildation(heart_rate=result["HeartRate"],
                                                          signal_to_noise_ratio=result["SNR"],
                                                          window_flag=(result["window"] == "signal passed"),
                                                          code="", final_code=3000)
    #  Convert the response to a dictionary
    signal_validation_result = json.loads(signal_validation_result)

    # Update the result dictionary with the validation response
    result["signal status"] = signal_validation_result["status"]
    result["message"] = signal_validation_result["response"]
    return result


def create_the_report(input_file, output_file, column_index=0, pdf_file="test.pdf"):
    # image_folder = "image_report"
    csv_file = "graph_output.csv"

    input_file = "graphs.csv"
    output_file = "graph_output.csv"

    # Read the input CSV file and exclude the desired column
    with open("signal_result.csv", "r") as file:
        reader = csv.reader(file)
        data = [row for row in reader]

    modified_data = []
    for row in data:
        modified_row = row[:column_index] + row[column_index+1:]
        modified_data.append(modified_row)

    # Write the modified data to the output CSV file
    with open("graphs.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(modified_data)

    row_index = 1  # index of the row to be removed

    # Read the input CSV file and exclude the desired row
    with open(input_file, "r") as file:
        reader = csv.reader(file)
        data = [row for row in reader if reader.line_num != row_index + 1]

    # Write the modified data to the output CSV file
    with open(output_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data)

    # Get a suitable font for increasing the font size
    font_prop = fm.FontProperties(size=14)

    # Read the CSV file into a DataFrame
    data = pd.read_csv(csv_file)

    # Create a PDF file for saving the merged output
    pdf_pages = PdfPages(pdf_file)

    # Iterate through the rows of the CSV file
    for _, row in data.iterrows():
        file_name = row['file_name']
        snr = row['SNR']
        status = row['Status']
        pantompkin = row['pantompkin']
        # message = row['message']
        heart_rate = row['HeartRate']
        window = row['window']
        image_path = row['image']
        Signal_status = row['signal status']

        # Check if the image file exists
        if os.path.isfile(image_path):
            # Read and process the image file
            image = plt.imread(image_path)

            # Create a new page in the PDF file
            fig = plt.figure(figsize=(8, 11))
            plt.axis('off')  # Remove x-axis and y-axis
            plt.imshow(image)

            # Create a table with the data
            table_data = [
                ['File Name:', file_name],
                ['SNR:', snr],
                ['Status:', status],
                ['Pantompkin:', pantompkin],
                # ['Message:', message],
                ['Heart Rate:', heart_rate],
                ['Window:', window],
                ['Signal Status:', Signal_status]
            ]

            # Add the CSV data as a table below the plot
            table = plt.table(cellText=table_data, loc='bottom',
                              cellLoc='left', colWidths=[0.3, 0.7])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            # Adjust the positioning of the graph and table
            # Adjust the rect values as needed
            fig.tight_layout(rect=[0, 0.25, 1, 1])

            # Set the properties of table cells to wrap the text
            for cell in table.get_celld().values():
                cell.set_text_props(multialignment='left', wrap=True)
                cell.set_height(0.1)  # Adjust the height value as needed

            # Add the plot to the PDF page
            pdf_pages.savefig()
            plt.close()

        else:
            print(f"Image file '{image_path}' not found. Skipping...")

    # Save and close the PDF file
    pdf_pages.close()


if __name__ == "__main__":
    # Replace with input CSV file path
    input_file = "signal_result.csv"
    # Replace with output CSV file path
    output_file = "graphs.csv"
    # Specify the column index to be removed
    column_index = 10

    # Process each file and generate results
    file_name = "data2/2390720_02_06_2023_11_2_56.json"
    process_data = process_the_signal(file_name)
    print(process_data)

    # # Create the report
    # create_the_report(input_file, output_file, column_index)
