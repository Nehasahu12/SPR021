# import numpy as np
# from scipy import signal

# import pywt  # type: ignore
# from scipy.signal import medfilt
# import matplotlib.pyplot as plt
# from collections import Counter
# from termcolor import colored  # type: ignore
# from sklearn.linear_model import LinearRegression  # type: ignore
# from utils.custom_print import print
# from scipy.fft import fft



import numpy as np
from scipy import signal

import pywt  # type: ignore
from scipy.signal import medfilt
import matplotlib.pyplot as plt
from collections import Counter
from termcolor import colored  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from utils.custom_print import print
from scipy.fft import fft


def get_num_samples(sample_rate, window_time):
    """
    Get the number of samples for a given time window, given a sampling rate.

    Args:
    sampling_rate (int or float): The sampling rate in Hz.
    window_time (int): The time window in milliseconds.

    Returns:
    int: The number of samples for the given time window.
    """
    # Convert window_time from milliseconds to seconds
    window_time_sec = window_time / 1000

    # Calculate the number of samples based on the sampling rate and time window
    num_samples = int(round(sample_rate * window_time_sec))

    return num_samples


def r_index_finder(qrs_segments, window_size=20, sample_rate=100):
    """
    This function takes QRS segments as input, squares each segment, finds the index of the maximum value, and identifies the R peak using a specified window size.

    Inputs:

    qrs_segments: a list or array of QRS segments
    window_size: an integer specifying the size of the window to search for the R peak before the maximum index value
    Outputs:

    r_peak_index: an integer representing the index of the R peak in the QRS segments
    Procedure:

    Square each QRS segment
    Find the index of the maximum value in the squared segments
    If the index of the maximum value is greater than zero, assign it as the R peak index
    If the index of the maximum value is zero or less, search for the maximum value index before the max_ind within the specified window size
    Assign the index of the maximum value before the max_ind as the R peak index
    Return the R peak index as output
    Example usage:
    qrs_segments = [0.1, 0.5, 0.8, 1.2, 1.5, 1.8, 1.2, 0.7, 0.2, -0.1, -0.5, -0.9, -0.6, -0.3, 0.1, 0.5, 0.9]
    window_size = 3
    r_peak_index = find_r_peak(qrs_segments, window_size)
    print(r_peak_index) # output: 5

    Note: This function assumes that the input QRS segments are in the correct order and are not missing any segments.
    """

    # Square The Segments
    sqr_qrs_seg = qrs_segments ** 2

    # finding maximum value index
    max_ind = np.argmax(sqr_qrs_seg)

    # Find value of max index
    max_ind_val = qrs_segments[max_ind]

    if max_ind_val > 0:
        # if max_ind_val > 0 then r peak will be max_ind
        r_peak = max_ind
        print("in function 74: No change")
    else:
        #  else, find maximum value index before max_ind within the given window size
        win_length = get_num_samples(sample_rate, window_size)  # in ms
        # print(f"in function 78: Win length: {win_length}, max ind: {max_ind}")
        if max_ind == 0:
            r_peak = 0
            # print("Reset Ero")
        else:
            stp = max_ind - win_length
            if stp <= 0:
                stp = 0
            # print(f"in function 78: sig length: {len(qrs_segments[stp: max_ind])}")
            # print(f"in function 78: sig length: {len(qrs_segments)}")
            r_peak = np.argmax(qrs_segments[stp: max_ind])

            r_peak = max_ind - win_length + r_peak
        #     _, r_peak = left_over(qrs_seg, max_ind, lth=0.01)
        # print("in function 81: change")

    return r_peak, qrs_segments[r_peak]

    
sampling_rate=76
def normalize_signal_range(dt, ll=-1, hl=1):
    """
    Normalize the input signal to a specified range.

    Args:
    - dt: a 1-D array-like object, representing the signal to be normalized
    - ll: a float, representing the lower limit of the normalized range (default: -1)
    - hl: a float, representing the upper limit of the normalized range (default: 1)

    Returns:
    - a 1-D array-like object with the same shape as dt, representing the normalized signal

    Example:
    >>> dt = [1, 2, 3, 4, 5]
    >>> normalize_signal_range(dt)
    array([-1., -0.5, 0., 0.5, 1.])
    """
    return hl*2 * (dt - min(dt)) / (max(dt) - min(dt)) - ll


def take_a_look_on_qrs(cleaned_signal, qrs_peak, sampling_rate=100, window_size=5, show=False):
    """
    win_length in millisecond
    :param window_size:
    :param qrs_peak:
    :param cleaned_signal:
    :type sampling_rate: object
    """

    r_peak = []
    r_peak_val = []

    win_size = int(get_num_samples(sampling_rate, window_size)/2)
    if show:
        plt.figure(figsize=(5, 5))

    for ind, val in enumerate(qrs_peak):
        start = val - win_size
        stop = val + win_size
        r_ind, r_val = r_index_finder(cleaned_signal[start:stop], window_size=5, sample_rate=sampling_rate)
        r_peak.append(start+r_ind)
        r_peak_val.append(r_val)

        if show:
            plt.title("In The take a look")
            plt.plot(cleaned_signal[start:stop])
            plt.scatter(r_ind, r_val)
    if show:
        plt.show()

    return np.array(r_peak), np.array(r_peak_val)


def max_occurrence(arr):
    # Create a dictionary to store the count of each element
    counts = {}
    for num in arr:
        if num in counts:
            counts[num] += 1
        else:
            counts[num] = 1

    # Find the element with the highest count
    max_count = 0
    max_num = None
    for num, count in counts.items():
        if count > max_count:
            max_count = count
            max_num = num

    return max_num


def calculate_fft(ecg_signal, sample_rate):
    # Perform FFT on the signal
    fft_signal = fft(ecg_signal)

    # Calculate the frequencies of the FFT result
    freqs = np.fft.fftfreq(len(ecg_signal), 1 / sample_rate)

    # Find the indices of the frequencies within the range of interest
    idx = np.where((freqs >= 0.5) & (freqs <= 30))

    return freqs[idx], np.abs(fft_signal[idx])


def adjacent_ratios(arr):
    arr = np.array(arr)
    return (arr[:-1] / arr[1:]).round(3)


def check_av_mis(lst, th=1.75):
    for ratio in lst:
        if ratio > th:
            return True
    return False


def rmse_1(predictions, targets):
    # return np.sqrt(np.mean((predictions - targets) ** 2))
    return predictions / targets


def stop_or_go(list1, list2, check_ind, window=10, th=20):
    max_allow_index = len(list1) - 1
    min_allow_index = 0
    initial_index = check_ind - window if check_ind - window > min_allow_index else check_ind
    final_index = check_ind + window if check_ind + window < max_allow_index else check_ind
    before_p = 1
    after_p = 1
    for li in range(initial_index, final_index):
        if li < check_ind:
            before_p += list1[li] - list2[li]
        if li > check_ind:
            after_p += list1[li] - list2[li]

    # print(f"Peak Index: {check_ind}\tBefore Index: {before_p}\tAfter Index: {after_p}\t{rmse_1(before_p, after_p)}")
    if rmse_1(before_p, after_p) <= th:
        return True
    return False


def find_first_closest_point(black_data, blue_data, peak_index, w):
    min_distance = float('inf')
    min_index = -1

    for abcd in range(peak_index, len(black_data)):
        # Define the start and end indices of the window around the current index i
        start_index = max(0, abcd - w)
        end_index = min(len(black_data), abcd + w + 1)

        # Loop over all points in the window and calculate the distance between the corresponding points
        for j in range(start_index, end_index):
            distance = abs(black_data[j] - blue_data[j])
            if distance < min_distance:
                min_distance = distance
                min_index = j

        if min_index != -1:
            # If the closest point is found within the window, return it
            return min_index

    # If no closest point is found, return -1
    return -1


def find_closest_points(list1, list2, th=10, min_distance=20):
    """
    :param min_distance:
    :param th:
    :param list1:
    :param list2:
    :return: Lowest distance between two list
    """
    # Find the index of the peak value in the first list
    peak_index = list1.index(max(list1))
    # Checking absolute maximum height of two list and swap Index if True
    # print(abs(max(list1)), min(list2))
    if abs(max(list1)) < abs(min(list2)):
        peak_index = list2.index(min(list2))
        # print("List Swapped ")

    # print(f"MinDistance: {min_distance}")
    min_distance += peak_index
    # print(f"MinDistance: {min_distance}")

    # print(f"Max value index: {peak_index}")
    prev = list1[peak_index] - list2[peak_index]
    l_index = peak_index
    all_closest = []
    all_closest_value = []
    tfth = max(list1) / 8
    try:
        for low_index, _ in enumerate(list1[peak_index:]):
            low_index = low_index + peak_index
            if abs(list1[low_index] - list2[low_index]) < prev:
                prev = abs(list1[low_index] - list2[low_index])
                l_index = low_index
                all_closest.append(l_index)
                all_closest_value.append(abs(list1[low_index] - list2[low_index]))

            if stop_or_go(list1, list2, low_index, th=th, window=5) and low_index >= min_distance and list1[low_index] < tfth:
                break

        # Use in case of single  QRS visualization
        min_index = all_closest[all_closest_value.index(min(all_closest_value))] + 5

        # # Use in case of Entire  QRS visualization
        # min_index = all_closest[all_closest_value.index(min(all_closest_value))] - peak_index

        # print(f"All Closed Index: {all_closest}, Index: {min_index + l_index}, Min index: {min_index}")
        # print(f"{before_p}\t{after_p}")
        return min_index, prev
    except Exception as e:
        # print(f"There is Error in closest points finder function: {e}")
        low_index, _ = right_deep(list1, peak_index, lth=None)
        return peak_index + low_index, list1[peak_index + low_index]


def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions - targets) ** 2))


def avg_finder(l1, l2):
    av = []
    for anv_index, j in zip(l1, l2):
        av.append((anv_index + j) / 2)
    return av


def qrs_end(ind_val, *args, **kwargs):
    val = [v[1] for v in ind_val]
    l_ind = [v[0] for v in ind_val]
    avg_gap = np.average(val)

    # Finding the lowest Root mean square error
    erl = rmse(avg_gap, val[0])
    org = val[0]
    for v in val:
        # print("RME: ", rmse(avg_gap, v))
        if rmse(avg_gap, v) < erl:
            erl = rmse(avg_gap, v)
            org = v
    # print(f"Nearest INdex: {val.index(org)}, Index: {l_ind[val.index(org)]}")
    return l_ind[val.index(org)]


def lowpass(raw_data, CUT_OFF_FREQUENCY=35, SAMPLING_FREQUENCY=100, ORDER=2):
    fs = 100  # Sampling frequency 1k
    fc = 35  # Cut-off frequency of the filter
    w = CUT_OFF_FREQUENCY / (SAMPLING_FREQUENCY / 2)  # Normalize the frequency
    b, a = signal.butter(ORDER, w, 'low')
    return signal.filtfilt(b, a, raw_data)


def highpass(raw_data, CUT_OFF_FREQUENCY=0.4, SAMPLING_FREQUENCY=100, ORDER=2):
    w = CUT_OFF_FREQUENCY / (SAMPLING_FREQUENCY / 2)  # Normalize the frequency
    b, a = signal.butter(ORDER, w, 'high')
    return signal.filtfilt(b, a, raw_data)


def notch_filter(sig, NOTCH_FREQUENCY=50, SAMPLING_FREQUENCY=100, QUALITY_FACTOR=10.0):
    # Design a notch filter using signal.iirnotch
    b_notch, a_notch = signal.iirnotch(NOTCH_FREQUENCY, QUALITY_FACTOR, SAMPLING_FREQUENCY)

    # Compute magnitude response of the designed filter
    freq, h = signal.freqz(b_notch, a_notch, fs=2 * np.pi)
    return signal.filtfilt(b_notch, a_notch, sig)


def bandpass(sig, FL=10, FH=20, SAMPLING_FREQUENCY=100, ORDER=2):
    return lowpass(highpass(sig, CUT_OFF_FREQUENCY=FL, SAMPLING_FREQUENCY=SAMPLING_FREQUENCY, ORDER=ORDER),
                   CUT_OFF_FREQUENCY=FH, SAMPLING_FREQUENCY=SAMPLING_FREQUENCY, ORDER=ORDER)


def derivative_filter(raw_data):
    return np.array([raw_data[i + 1] - raw_data[i] for i in range(len(raw_data) - 1)])


def integration(derivative, init=0):
    """
    parameter: derivative of the signal
    returns: integration of the signal
    """
    sout = []
    for i in range(len(derivative)):
        sout.append(init + derivative[i])
        init = init + derivative[i]
    return np.array(sout)


def integration_percent(derivative, init=0, per=1):
    """
    parameter: derivative of the signal
    returns: integration of the signal
    """
    sout = []
    for i in range(len(derivative)):
        sout.append((init + derivative[i]) * per)
        init = init + derivative[i]
    sout.append(per * (init + derivative[i]))
    return np.array(sout)


def best_fit(sts):
    x = np.arange(len(sts)).reshape((len(sts), 1))
    regr = LinearRegression()
    regr.fit(x, sts)
    y_pred = regr.predict(x)
    return sts - y_pred


def second_derivative_filter(raw_data):
    return np.array([raw_data[i + 2] - raw_data[i] for i in range(len(raw_data) - 2)])


def third_derivative_filter(raw_data):
    return np.array([raw_data[i + 3] - raw_data[i] for i in range(len(raw_data) - 3)])


def forth_derivative_filter(raw_data):
    return np.array([raw_data[i + 4] - raw_data[i] for i in range(len(raw_data) - 4)])


def nth_derivative_filter(raw_data, n):
    return np.array([raw_data[i + n] - raw_data[i] for i in range(len(raw_data) - n)])


def filter_sub_signal(sbs, frq=15):
    temp_lst = [highpass(lowpass(sub_div, CUT_OFF_FREQUENCY=frq), CUT_OFF_FREQUENCY=0.5) for sub_div in sbs]
    return np.array(temp_lst)


# TODO: Find absolute percentage difference between the two signals/values
def slop_percent(x1, x2):
    return 100 * (abs((x2 - x1) / x1)) if x1 != 0 else 100 * (abs((x2 - x1) / 1))


def lowest_slop_finder(xy):
    slp = []
    for i in range(len(xy) - 1):
        a = slop_percent(xy[i], xy[i + 1])
        slp.append(a)
    return np.array(slp).argmin()


def find_phase(normalized_ecg, Rv=0.001):
    return np.arctan(normalized_ecg / Rv)


# Moving Average Filter
def sub_moving_average(abc, n=10):
    temp_abc = []
    for a in abc:
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        temp_abc.append(ret[n - 1:] / n)
    return np.array(temp_abc)


def yusuf(sts):
    pp = signal.find_peaks(sts)
    amp = [sts[i] for i in pp[0]]
    cp = abs(sum(amp))
    pp = signal.find_peaks(-sts)
    amp = [sts[i] for i in pp[0]]
    cn = abs(sum(amp))

    return cp > cn


def get_convoled_signal(filtered_data, response):
    """
    Param: Filtered 2D Ecg Signal, Transfer function
    Return: 2D Array after Convolution of Data
    """
    temp_lst = [np.convolve(fsd, response) for fsd in filtered_data]
    return np.array(temp_lst)


# Dectect The ECG Signal is Lower or Upper for the given ECG Signal
def detect_signal_class(Lead_data):
    """
    parameter: ECG Signal Array
    return: False if ECG Signal is Lower, True if ECG Signal is Upper
    """
    base = Lead_data.mean()
    lower = Lead_data.min()
    upper = Lead_data.max()

    # Calculate the difference between from the Base
    d1 = abs(upper - base)
    d2 = abs(lower - base)
    if d1 > d2:
        # print("Upper Signal")
        return True
    else:
        # print("Lower Signal")
        return False


# Split the data into chunks
def split_ecg_signal_p(p, index, left_m=0, right_m=0):
    if len(index) >= 2:
        sub_ecg_signal = []
        for i in range(len(index) - 2):
            initial = (int((index[i + 1] - index[i]) / 2) + index[i]) - left_m
            final = (index[i + 1] + int((index[i + 2] - index[i + 1]) / 2)) + right_m
            #             print("Index : {}, {} and sample= {}".format(initial, final, final-initial))
            sub_ecg_signal.append(p[initial:final])
        return sub_ecg_signal


def segmentation(p, index, win=150, margin=0):
    sub_ecg_signal = []
    if len(index) >= 2:
        win = win + margin
        for ind, val in enumerate(index):
            start = val - win
            stop = val + win
            if start < 0: start = 0
            if stop > len(p): stop = len(p)
            sub_ecg_signal.append(p[start:stop])

    return sub_ecg_signal


def split_ecg_signal(ecg_signal, window):
    """
    param: ecg_signal - the signal to split
            window - the window list of [lower_limit, upper_limit]
    """
    temp_list = []
    sth = 0
    for l, u in window:
        if l < 0:
            l = 0
            sth = 1
        if u > len(ecg_signal):
            u = len(ecg_signal) - 1
            sth = 1
        if sth == 0:
            temp_list.append(ecg_signal[l:u])
        sth = 0
    #         print(l, u)

    return np.array(temp_list)


def qrs_signal_finder(ecg_new, peak_index, margin=0, sdur=5, qdur=4, isoelectric=0):
    # sourcery skip: low-code-quality
    """
    param: 1D Ecg Signal, R peak Index,
    margin: 0
    Using Top-Down approach
    """
    flg = peak_index
    lec = []
    index = []
    index2 = []
    base_line_wonder = ecg_new.mean()
    pointer = 0  # Pointer to counter number of sample Before R peak Index
    # Trverse the signal from the peak to the left baseline
    while True:
        #         Top To down travel through left side of the signal
        if (ecg_new[flg] - ecg_new[flg - 1]) > -1 * margin or ecg_new[flg] > 0.01:
            #         print(ecg_new[flg])
            lec.append(ecg_new[flg])
            index.append(flg)

            if flg <= 2:
                break

            # if number of sample is greater than the window size then break
            if pointer > 60:
                # print("There is too much sample before the R peak")
                break
            flg -= 1
            pointer += 1
        else:
            break

    Q_point = flg
    q_cont = 0
    # Traverse the signal from lower to upper in left direction
    while True:
        #         print("Second: ", (one_ecg_plus[flg]-one_ecg_plus[flg-1]))
        #         Down to Top travel through left side of the signal
        if (ecg_new[flg] - ecg_new[flg - 1]) < 0 and ecg_new[flg] <= base_line_wonder:
            #         print(ecg_new[flg])
            lec.append(ecg_new[flg])
            index.append(flg)
            #         print(one_ecg_plus[flg])
            if flg <= 2:
                break
            flg -= 1
            q_cont += 1
        else:
            break

        if q_cont > qdur:
            break

    flg = peak_index
    rec = []
    # Traverse the signal from the peak to the right base line
    while True:
        #         print(flg)
        if (ecg_new[flg] - ecg_new[flg + 1]) >= -1 * margin:
            #         print(ecg_new[flg])
            rec.append(ecg_new[flg])
            index2.append(flg)
            if flg >= len(ecg_new) - 2:
                break
            flg += 1
        else:
            break

    S_point = flg
    # Go from s point to s_offset point
    pc = 0
    s_off = []
    ind3 = []
    # Traverse the signal from lower to upper in right direction
    # print("-------------------------------------")
    # print("Len: ", len(ecg_new))
    while True:
        # print("[-->>}]", slop_percent(ecg_new[flg], ecg_new[flg+1]), "\t", flg)
        if (ecg_new[flg] - ecg_new[flg + 1]) < 0 and ecg_new[flg] <= base_line_wonder:
            # print("S_OFF: ", ecg_new[flg], flg)
            ind3.append(flg)
            s_off.append(ecg_new[flg])
            flg += 1
            pc += 1
            if flg >= len(ecg_new) - 1:
                # print("Break", flg)
                break
        else:
            # print("---------")
            break

        if pc > sdur:
            break

    # Finding Lowwest slop change
    bs_signal = ecg_new[S_point + 2:flg]
    if len(bs_signal) > 2:
        # print("Bs Signal:", bs_signal)
        # print("Diff: ", np.diff(bs_signal))
        # print("Index: ", ind3)
        min_slop = np.diff(bs_signal).argmin() + 2
        # print("Min Slope: ", min_slop)
        ind3 = ind3[:min_slop + 2]
        s_off = s_off[:min_slop + 2]
        # print("Index 3: ", ind3)
        # print("==========================================")

    s_offset = flg
    #     print("Q: {}, S: {}".format(Q_point, S_point))
    lec.reverse()
    index.reverse()
    index = index + index2 + ind3
    #     qrs_data = (lec+rec+new_sampl)
    qrs_data = (lec + rec + s_off)
    return [qrs_data, index, Q_point, S_point, s_offset]


# -----------31-08-2021--->--27-05-2022------
def remove_ramp(t, order=3):
    """
    parm: list of data
    Return: retun signal after removing ramp from signal
    """
    x = range(0, len(t))
    order = np.polyfit(x, t, order)
    tt = np.polyval(order, x)
    mi = t.min()
    return (t - tt) + mi


# Split the data into chunks
def signal_segmentation(p, index):
    sub_ecg_signal = []
    if len(index) >= 2:
        for i in range(len(index) - 2):
            initial = int((index[i + 1] - index[i]) / 2) + index[i]
            final = index[i + 1] + int((index[i + 2] - index[i + 1]) / 2)
            #             print("Index : {}, {} and sample= {}".format(initial, final, final-initial))
            sub_ecg_signal.append(p[initial:final])
        return sub_ecg_signal


def left_deep(sg, peak, lth=None, margin=1, slop=0.0005):
    """
    param: lth=Threshold, margin, Slope
    Return: [nomber_of_sample, index]
    """
    # print(peak)
    m = 0
    while True:
        # print("----------------------")
        temp_index = peak
        index_counter = 0
        if lth is None:
            while True:
                # print(peak)
                # if (sg[temp_index] > sg[temp_index - 1] or sg[temp_index] - sg[temp_index - 1] > -1 * slop) and temp_index < len(sg)-2:
                if sg[temp_index] > sg[temp_index - 1] and temp_index < len(sg) - 2:
                    index_counter += 1
                    temp_index -= 1
                    # print(temp_index)
                    if temp_index <= 0:
                        break

                else:
                    break
            # print("threshold None", sg[temp_index] - sg[peak])
            # print(index_counter)
            temp_index = 0 if temp_index < 0 else temp_index
            # return index_counter, temp_index
        else:
            while True:
                # print("Temping: ", temp_index)
                if sg[temp_index] >= lth and temp_index > 0:
                    index_counter += 1
                    temp_index -= 1
                else:
                    break
            # print(sg[temp_index] - sg[peak])
            # print(index_counter)
            temp_index = 0 if temp_index < 0 else temp_index

        if index_counter > 0:
            # print("Left Deep signing out")
            return index_counter, temp_index
        else:
            # print("Correcting the peak")
            peak -= 1
            if m > margin:
                return 0, 0
            m += 1

            if peak < 0:
                return 0, 0


def right_deep(sg, peak, lth=None, margin=1, slop=0.0005):
    """
    Return: [number_of_sample, index]
    """
    m = 0
    while True:
        temp_index = peak
        index_counter = 0
        if lth is None:
            while temp_index < len(sg) - 2:
                if sg[temp_index] > sg[temp_index + 1]:
                    index_counter += 1
                    temp_index += 1
                else:
                    break

        # return index_counter, temp_index
        else:
            while True:
                if sg[temp_index] > lth and temp_index < len(sg) - 2:
                    index_counter += 1
                    temp_index += 1
                else:
                    break
        #     print(sg[temp_index] - sg[peak])
        #     print(index_counter)
        # return index_counter, temp_index
        if index_counter > 0:
            # print("Right Deep signing out")
            return index_counter, temp_index
        else:
            peak += 1
            if m > margin:
                return 0, 0
            m += 1
            if peak > len(sg) - 2:
                return 0, 0


def left_over(sg, peak, lth=None, slop=0.0025, win=False, maxtime=60, prst=0.0025):
    """
    Return: [number_of_sample, index]
    """
    #     print(peak)
    margin_prev_sam = int((maxtime / 1000) / prst)
    # print(f"margin prev sample limit: {margin_prev_sam}")
    temp_index = peak
    index_counter = 0
    # print("Type of Signal: ", type(sg))
    if lth is None:
        # print("Temp index: ", temp_index)
        while True:
            if temp_index != 0:
                if sg[temp_index] < sg[temp_index - 1] and temp_index > 0:
                    # len(sg)-2:
                    index_counter += 1
                    temp_index -= 1
                    if index_counter > margin_prev_sam and win: break
                else:
                    break
            else:
                break
        #     print(sg[temp_index] - sg[peak])
        #     print(index_counter)
        # print("Temp: ", temp_index)
        temp_index = 0 if temp_index < 0 else temp_index
        return index_counter, temp_index
    else:
        while True:
            if sg[temp_index] <= lth and temp_index > 0:  # len(sg)-2:
                index_counter += 1
                temp_index -= 1
                if index_counter > margin_prev_sam and win: break
            else:
                break
        #     print(sg[temp_index] - sg[peak])
        #     print(index_counter)
        # print("Temp: ", temp_index)
        temp_index = 0 if temp_index < 0 else temp_index
        return index_counter, temp_index


def right_over(sg, peak, lth=None, slop=0.0025):
    """
    Return: [number_of_sample, index]
    """
    temp_index = peak
    index_counter = 0
    if lth is None:
        while True:
            if sg[temp_index] < sg[temp_index + 1] and temp_index < len(sg) - 2:
                index_counter += 1
                temp_index += 1
            else:
                break
        #     print(sg[temp_index] - sg[peak])
        #     print(index_counter)
        return index_counter, temp_index
    else:
        while True:
            if sg[temp_index] <= lth and temp_index < len(sg) - 2:
                index_counter += 1
                temp_index += 1
            else:
                break
        #     print(sg[temp_index] - sg[peak])
        #     print(index_counter)
        return index_counter, temp_index


# J point Detector
def j_point_detector(dt, st_p, slope=0.001, show=False):
    """
    param: dt, st_p, slope
    Return: J point index, value, number of sample if there first vaild slop, else return 20th sample from the begining
    """
    m = slope
    up = []
    up_v = []
    down = []
    down_v = []

    uc = 0
    dc = 0
    nc = 0
    j_count = 0
    # print(dt)
    # print("STP - " + str(st_p) + " " + str(slope))
    # print(st_p + lowest_slop_finder(dt[5:]))

    for n in range(len(dt) - 4):
        f1 = dt[n]
        f2 = dt[n + 1]
        f3 = dt[n + 2]
        f4 = dt[n + 3]
        if f1 < f2 and f1 < f3 and f1 < f4:
            df1 = f2 - f1
            df2 = f3 - f1
            df3 = f4 - f1
            #         print(df1,"->", df2,"->", df3)
            if df1 > m and df2 > m and df3 > m:
                up.append(n)
                up_v.append(f1)
                uc += 1
                # print("Upside")
                pass

        elif f1 > f2 and f1 > f3 and f1 > f4:
            df1 = -f2 + f1
            df2 = -f3 + f1
            df3 = -f4 + f1
            #         print(df1,"->", df2,"->", df3)
            if df1 > m and df2 > m and df3 > m:
                down.append(n)
                down_v.append(f1)
                dc += 1
                # print("Downside")
                pass

        else:
            return [st_p + n + 3, f4, n + 3]

        # Return When J point finder Index is exceed the length of 20,
        if j_count > 20:
            return [st_p + n + 3, f4, n + 3]
        j_count += 1


# diagnostic
def denoise_signal(X, dwt_transform, dlevels, cutoff_low, cutoff_high):
    coeffs = pywt.wavedec(X, dwt_transform, level=dlevels)  # wavelet transform 'bior4.4'
    # scale 0 to cutoff_low
    for ca in range(0, cutoff_low):
        coeffs[ca] = np.multiply(coeffs[ca], [0.0])

    # scale cutoff_high to end
    for ca in range(cutoff_high, len(coeffs)):
        coeffs[ca] = np.multiply(coeffs[ca], [0.0])
    Y = pywt.waverec(coeffs, dwt_transform)  # inverse wavelet transform
    return Y




# def get_median_filter_width(sampling_rate, duration):
#     res = int( sampling_rate*duration )
#     res += ((res%2) - 1) # needs to be an odd number
#     return res

#     # baseline fitting by filtering
#     # === Define Filtering Params for Baseline fitting Leads======================
#     ms_flt_array = [0.2,0.6]    #<-- length of baseline fitting filters (in seconds)
#     mfa = np.zeros(len(ms_flt_array), dtype='int')
#     for i in range(0, len(ms_flt_array)):
#         mfa[i] = get_median_filter_width(sampling_rate, ms_flt_array[i])

def filter_signal(X):
    global mfa
    X0 = X  # read orignal signal
    for mi in range(0, len(mfa)):
        X0 = medfilt(X0, mfa[mi])  # apply median filter one by one on top of each other
    X0 = np.subtract(X, X0)  # finally subtract from orignal signal
    return X0





def get_median_filter_width(sampling_rate, duration):
    res = int(sampling_rate * duration)
    res += ((res % 2) - 1)  # needs to be an odd number
    return res


# baseline fitting by filtering
# === Define Filtering Params for Baseline fitting Leads======================
ms_flt_array = [0.2, 0.6]  # <-- length of baseline fitting filters (in seconds)
mfa = np.zeros(len(ms_flt_array), dtype='int')
for i in range(0, len(ms_flt_array)):
    mfa[i] = get_median_filter_width(sampling_rate, ms_flt_array[i])


def cross_finder(line, uper_p, lower_p):
    app = np.concatenate((uper_p, lower_p))
    app.sort()
    #     print(app)
    cross_in = []
    for i in range(len(app) - 1):
        initial = app[i]
        final = app[i + 1]
        if line[initial] >= line[final]:
            for j in range(initial, final):
                if line[j] >= 0:
                    pass
                else:
                    #                 print(j, end=" -> ")
                    cross_in.append(j)
                    break
        else:
            for j in range(initial, final):
                if line[j] <= 0:
                    pass
                else:
                    #                 print(j, end=" -> ")
                    cross_in.append(j - 1)
                    break
    return cross_in


def r_peak_finder(dt, ma, mi):
    mm = zip(ma, mi)
    local_slop = 0
    r_index = []
    r_value = []
    for local_max, local_min in mm:
        #          Check To ward the Maximum peak
        if local_min > local_max:
            # print("Normal QRS complex")
            local_inex = local_max
            center_point = dt[local_max] - dt[local_min]
            while True:
                if dt[local_inex] >= 0:
                    local_inex += 1
                else:
                    break
            r_index.append(local_inex - 1)
            r_value.append(dt[local_inex - 1])
        else:
            # print("large negative QRS deflection")
            local_inex = local_min
            center_point = dt[local_min] - dt[local_max]
            while True:
                if dt[local_inex] <= 0:
                    local_inex += 1
                else:
                    break
            r_index.append(local_inex)
            r_value.append(dt[local_inex])
    return r_index, r_value


# Find center point between two peaks
def find_center_point(local_sig, local_peak):
    """
    param local_sig: local signal
          local_peak: local peak

    return: center point
    Description: Find center point between two peaks
    """
    center_peak = []
    center_peak_value = []
    for local_ind in range(len(local_peak) - 1):
        center_point = int((local_peak[local_ind + 1] - local_peak[local_ind]) / 2) + local_peak[local_ind]
        if center_point >= len(local_sig):
            center_point = len(local_sig) - 2
        center_peak.append(center_point)
        center_peak_value.append(local_sig[center_point])

    return [center_peak, center_peak_value]


# Filter peaks based on the largest positive peak or negative peak
def filter_peaks(local_sig, local_peak):
    if len(local_peak) == 1:
        return local_peak

    local_peak_index = local_peak[0]
    local_peak_value = local_sig[local_peak_index]

    for i in range(1, len(local_peak)):
        if local_sig[local_peak[i]] > local_peak_value:
            local_peak_index = local_peak[i]
            local_peak_value = local_sig[local_peak_index]
        else:
            pass

    return local_peak_index


# -------------Date : 07/07/2022 ---------------

# Function for convolution with ricker function and return result
def maximize_qrs_signal(oroginal, length=30, sigma=2, mode='same', max_height=True):
    dur = np.linspace(0, np.pi, 20)
    temp1 = np.sin(dur)
    if max_height: return np.convolve(np.convolve(oroginal, signal.ricker(length, sigma), mode=mode), temp1,
                                      mode=mode)  # type: ignore
    return np.convolve(oroginal, signal.ricker(length, sigma), mode=mode)  # type: ignore


# function to find the baseline of the signal
def find_baseline(x, order=1, zeros=False):
    """
    param x: signal, order: order of the polynomial, zeros: if True, zeros based baseline is used
    return: array of baseline values
    """
    y = range(len(x))
    coff = np.polyfit(y, x, order)
    if not zeros:
        return np.polyval(coff, y)
    else:
        return np.zeros(len(x))


# function to find the threshold for the signal
def find_threshold(x, div_factor=10.0):
    """
    param x: signal, div_factor: factor to divide the maximum value of the signal
    return: threshold value
    """
    if div_factor == 0: div_factor = 1
    return (x.max() - x.min()) / div_factor


def find_correct_qrs_peaks(ecg_array, temp_qrs_indices, window_size=20, sampling_rate=100):
    """
    This function takes as input an ECG array, an array of temporary QRS indices,
    a window size in milliseconds (default value is 40), and a sampling rate in Hz (default value is 500).
    It returns two arrays containing the indices and values of the corrected QRS peaks.

    :param ecg_array: An array of ECG data
    :param temp_qrs_indices: An array of temporary QRS indices
    :param window_size: The size of the window in milliseconds (default value is 40)
    :param sampling_rate: The sampling rate in Hz (default value is 500)
    :return: Two arrays containing the indices and values of the corrected QRS peaks
    """
    corrected_qrs_peak_indices = []
    corrected_qrs_peak_values = []

    for temp_qrs_index in temp_qrs_indices:
        # 1. find number of sample based on window_size and temp_qrs_index
        num_samples = int((window_size / 1000) * sampling_rate)
        lower_limit = max(0, temp_qrs_index - num_samples)
        upper_limit = min(len(ecg_array), temp_qrs_index + num_samples)

        # 2. split ecg_array with the help of lower and upper limit
        temp_array = ecg_array[lower_limit:upper_limit]

        # 3. find the index and value of first largest number in array
        ind_1 = temp_array.argmax()
        val_1 = temp_array[ind_1]

        # 4. find the index and value of second-largest number in array
        temp_array[ind_1] = -float('inf')
        ind_2 = temp_array.argmax()
        val_2 = temp_array[ind_2]

        # 5. check conditions to determine corrected_qrs_peak
        if val_1 > val_2 and ind_1 < ind_2:
            corrected_qrs_peak_index = ind_1
            corrected_qrs_peak_value = val_1
        else:
            corrected_qrs_peak_index = ind_2
            corrected_qrs_peak_value = val_2

        # add corrected qrs peak index and value to respective lists
        corrected_qrs_peak_indices.append(corrected_qrs_peak_index + lower_limit)
        corrected_qrs_peak_values.append(corrected_qrs_peak_value)

    # return lists of corrected qrs peak indices and values
    return corrected_qrs_peak_indices, corrected_qrs_peak_values


# function for correction of the signal peak
def correct_qrs_peaks(sig, peak_index, index_amplitude):
    """
    param sig: signal, peak_index: peak index, index_amplitude: amplitude of the peak
    return: list of corrected peak index and amplitude ex: [index, amplitude]
    """

    nd = []
    aam = []
    lda = []
    rua = []
    prev = 0
    # print("Index amplitude: {}".format(index_amplitude))
    # print("-------------------------")
    for ind, _ in zip(peak_index, index_amplitude):
        # Correcting the peak with the basis of the nearest peak

        # right_error = right_over(sig, ind)
        # left_error = left_over(sig, ind)
        # print(f"Actual index: {ind} Right Error: {ind + right_error[0]} and Left Error: {ind + left_error[0]}")
        new_index = ind + right_over(sig, ind)[0] - left_over(sig, ind)[0]
        new_value = np.array(sig[new_index])

        nd.append(new_index)
        aam.append(new_value)

        # # New Line of Code Added on 17-10-2022 ===========
        #
        # # Index
        # ld, ln = left_deep(sig, new_index, slop=0)
        # rd, rn = right_deep(sig, new_index, slop=0)
        #
        # # print("{}, left_deep: {}, right deep: {}".format(new_index, ln, rn))
        #
        # sp = new_index - ld
        # rp = new_index + rd
        # qp = sp - left_over(sig, sp)[0]
        # if ln <= 4:
        #     new_index = qp
        #     new_value = np.array(sig[new_index])
        # if rn <= 10:
        #     new_index = qp
        #     new_value = np.array(sig[new_index])
        #
        # # Amplitude
        # sa = sig[sp]
        # qa = sig[qp]
        # ra = sig[rp]
        #
        # # Distance
        # l1 = new_value - sa
        # l2 = qa - sa
        # l3 = new_value - ra
        #
        # # print(new_index,  "******") print("L1: ", sp, l1, "L2: ", qp, l2, "\t", l1, l2, l3, "\t", "Condition: ",
        # # l1 > l2, "\tIncorrect: ", rn==0) if True else False
        #
        # per = round(slop_percent(l1, l2), 3)
        # # print("Ratio: {}".format(per))
        #
        # # Change R peak to the left peak if ration is less than 50%
        # if per < 50:
        #     new_index = qp
        #     new_value = np.array(sig[new_index])
        #
        # # if new_index == 0:
        # #     _, _ = right_over(sig, new_index)
        #
        # # ----------------------------------
        # lda.append(sig[sp])
        # rua.append(sig[sp - left_over(sig, sp)[0]])
        # # ==============================================
        #
        # # print("New index: {}".format(new_index))
        # if new_index > 5:
        #     nd.append(new_index)
        #     aam.append(new_value)

    # print("correct Index amplitude: {}".format([int(i) for i in aam]))
    # print("left Deep Index amplitude: {}".format(lda))
    # print("left Over Index amplitude: {} \n\n".format(rua))
    # print(len(nd))
    # print(len(aam))
    # print("++++++++++++++++")
    return np.array(nd), np.array(aam)


def area_under_curve(sig):
    # print("Yusuf : ", yusuf(sig))
    s = sum(sig)
    # print("AUC", s)
    if s > 0:
        return True
    else:
        return False


# function for corection of the signal peak
def correct_diff_qrs_peaks(sig, peak_index, index_amplitude):
    """
    param sig: signal, peak_index: peak index, index_amplitude: amplitude of the peak
    return: list of corrected peak index and amplitude ex: [index, amplitude]
    """

    nd = []
    aam = []
    # print("Index amplitude: {}".format(index_amplitude))
    for ind, _ in zip(peak_index, index_amplitude):
        # Correcting the peak with the basis of nearest peak

        # right_error = right_over(sig, ind)
        # left_error = left_over(sig, ind)
        new_index = ind + right_over(sig, ind)[0] - left_over(sig, ind)[0]
        new_value = np.array(sig[new_index])
        if new_index > 5:
            nd.append(new_index)
            aam.append(new_value)
    # print("in the Function: ", len(nd), len(aam))
    return np.array(nd), np.array(aam)


def j_point_finder(cleaned_signal, win_legth=4, qrs_peak=None, show=False):
    qrs_threshold = 8.0
    qrs_distance = 100

    win = np.linspace(-1, 1, win_legth)
    plt.grid()
    maximized_signal = np.convolve(cleaned_signal, win, mode='same')
    threshold = find_threshold(maximized_signal, div_factor=qrs_threshold)
    if qrs_peak is None:
        # print("Threshold : ", threshold)
        qrs_peaks = signal.find_peaks(maximized_signal, height=threshold, distance=qrs_distance)
        peak_apmlitude = np.array(maximized_signal[qrs_peaks[0]])
        # print("QRS Peak: ", qrs_peaks)

        # Correct peaks of QRS
        corrected_peaks = correct_diff_qrs_peaks(maximized_signal, qrs_peaks[0], peak_apmlitude)
        # print("peak correction: ", len(corrected_peaks[0]), len(peak_apmlitude))
        # qrs_peaks = corrected_peaks
        # print("Corrected peaks of QRS: ", corrected_peaks)
    else:
        maximized_signal = cleaned_signal
        corrected_peaks = [qrs_peak, [cleaned_signal[i] for i in qrs_peak]]
        # print(">> Corrected peaks of QRS: ", corrected_peaks)

    j_p = []
    j_a = []
    j_p2 = []
    j_a2 = []
    sub = int(win_legth / 2)
    # print("===>", qrs_peak)
    # for i in corrected_peaks[0]:
    signal_length = len(cleaned_signal)
    for i in qrs_peak:
        _, dsi = right_deep(maximized_signal, i, lth=0)
        _, usi = right_over(maximized_signal, dsi, lth=0)
        usi -= sub
        j_p.append(usi)
        j_a.append(cleaned_signal[usi])

        tang = find_tangent_line(cleaned_signal[dsi:usi], t_peak=usi, prev_n=60)
        # print(f"Length of Cleaned Signal: {len(cleaned_signal)}")
        if tang[0][-1] > i:
            if tang[0][-1] > signal_length:
                j_p2.append(signal_length - 1)
                j_a2.append(cleaned_signal[signal_length - 1])
            else:
                j_p2.append(tang[0][-1] - 1)
                j_a2.append(cleaned_signal[tang[0][-1] - 1])
        else:
            j_p2.append(usi)
            j_a2.append(cleaned_signal[usi])

        # print(tang)
        # for i in tang:
        #     plt.plot(i[0], i[1])

    if show:
        # print(maximized_signal)
        plt.title("J points o")
        plt.plot(range(len(maximized_signal)), maximized_signal)
        plt.plot(range(len(cleaned_signal)), cleaned_signal)
        plt.scatter(corrected_peaks[0], corrected_peaks[1], color='black')
        plt.scatter(j_p, j_a, color="red", s=100)
        plt.scatter(j_p2, j_a2, color="green", s=100)
        plt.axhline(threshold, color="red")
        plt.legend(["Convolution", "original"])
        plt.show()
    return j_p2


# ================Date: 08/07/2022====================
# Calculate basic parameters of the signal based on peak
def calculate_basic_parameters(x, peaks):
    return "Need to implements this function"


# Moving average filter
def moving_average(a, n=3):
    """Return moving average of array a."""
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


# ============Date: 09/07/2022====================

# Find Baseline of the signal with respect to the peak and data between the T-offset to P-Onset
def baseline_finder(x, peaks, margin=10, mode=None):
    """Return baseline of the signal with respect to the peak and data between the T-offset to P-Onset
    param x: signal, peaks: peak index, margin: margin between T-offset to P-Onset
    return: [[baseline_index], [baseline_amplitude], [2nd_base_amplitude], overall_baseline]
    """
    # print(f"Peaks: {peaks[0]}")
    tp = find_center_point(x, peaks[0])
    # print(f"TP: ", tp)
    if mode is not None:
        avg_baseline = np.average(tp[1])
        return avg_baseline
    dx = np.diff(x)
    da = [dx[i] for i in tp[0]]
    dy = [o - b for o, b in zip(tp[1], da)]
    # print(tp[0])
    # print(tp[1])
    new_th = []
    for index in tp[0]:
        if index - margin > 0 and index + margin < len(x):
            ln, _ = left_deep(x, index, slop=0)
            rn, _ = right_deep(x, index, slop=0)
            # print("INDEX: ", index)
            # print("INDEX: ",ln, rn, index)
            if ln > rn:
                index -= ln
            else:
                index += rn
            # print(index, index-margin, index+margin)
            avc = x[index - margin:index + margin].mean()
            # print("Centered on ", avc)
            new_th.append(avc)
        else:
            ln, _ = left_deep(x, index)
            rn, _ = right_deep(x, index)
            # print(ln, lindx, rn, rindx)
            if ln > rn:
                index -= ln
            else:
                index += rn
        # print("New index: ", index, x[index])
        new_th.append(x[index])

    tp.append(new_th)
    tp.append(np.average(dy))
    tp.append(np.average(new_th))
    # print("AVR Average MEan: ", tp)
    # print("[::] Average Baseline: ", np.average(new_th))
    return tp


# ==========Date: 12/10/2022====================
def baseline_ml_finder(x, peaks, margin=10):
    tp = find_center_point(x, peaks[0])
    new_th = []
    # print("Peaks: ", peaks)
    for index in tp[0]:
        if index - margin > 0 and index + margin < len(x):
            ln, _ = left_deep(x, index, slop=0)
            rn, _ = right_deep(x, index, slop=0)
            # print(ln,  rn, index)
            if ln > rn:
                index -= ln
            else:
                index += rn
            # print(index, index-margin, index+margin)
            avc = x[index - margin:index + margin].mean()
            # print("Centered on ", avc)
            new_th.append(avc)
        else:
            ln, _ = left_deep(x, index)
            rn, _ = right_deep(x, index)
            # print(ln, lindx, rn, rindx)
            if ln > rn:
                index -= ln
            else:
                index += rn
            # print("New index: ", index)
        new_th.append(x[index])

    # Creating a linear regression model
    reg = LinearRegression()
    x = np.arange(len(new_th)).reshape((-1, 1))
    # print("function.py 1194 x = ", x)
    # print("ABCDEFGH: ", len(x), len(new_th))
    if len(x) == len(new_th) and len(x) > 0:
        reg.fit(x, new_th)
        # baseline prediction
        # pared = reg.predict(x)
        tp.append(new_th)
        tp.append(reg.intercept_)
    else:
        tp.append(new_th)
        tp.append(0)

    return tp


# Function for finding the tangential slope
def find_tangent_line(y, prev_n=50, baseline=0, t_peak=0, baseline_stop=True):
    """
    parameter: y - input Array
                prev_n - number of points to follows
                baseline - isoelectric_line
    Returns: x, y_line
    """
    # print(f"Length of signal: {len(y)} points")
    # print(f"Baseline: {baseline} points")s
    if len(y) >= 2:
        # finding x[n+1] - x[n] and maximum value index
        ind = (np.diff(y) ** 2).argmax()
        slop = y[ind] - y[ind + 1]
        initial = y[ind] + (slop * prev_n)
        line = []
        x = []
        if initial >= 0:
            for i in range(prev_n * 2):
                line.append(initial)
                x.append(t_peak + ind + i - prev_n)
                if initial < baseline and baseline_stop:
                    break
                initial -= slop
            return x, line
        else:
            for i in range(prev_n * 2):
                line.append(initial)
                x.append(t_peak + ind + i - prev_n)
                if initial > baseline and baseline_stop:
                    break
                initial -= slop
            return x, line
    else:
        return [0, 0], [0, 0]


# Function for finding the tangential slope
def find_tangent_p_onset(y, prev_n=5, baseline=0, t_peak=0, baseline_stop=True):
    """
    parameter: y - input Array
                prev_n - number of points to follows
                baseline - isoelectric_line
    Returns: x, y_line
    """
    try:
        # print(f"Length of signal: {len(y)} points", t_peak)
        if len(y) >= 2:
            # finding x[n+1] - x[n] and maximum value index
            # print("HLP: ", y[0:t_peak])
            if t_peak > 0:
                ind = (np.diff(y[0:t_peak]) ** 2).argmax()
            else:
                ind = (np.diff(y) ** 2).argmax()
            slop = y[ind + 1] - y[ind]
            initial_t = y[ind] + (slop * prev_n)
            #         print("Meta: ", initial, slop, ind)
            line = []
            x = []
            if initial_t >= 0:
                # print("Greater:")
                for i in range(prev_n * 2):
                    line.append(initial_t)
                    #                 print(t_peak-i, initial)
                    x.append(t_peak - i - 1)
                    if initial_t < baseline and baseline_stop:
                        break
                    initial_t -= slop
                return x, line
            else:
                # print("Lesser:")
                for i in range(prev_n * 2):
                    line.append(initial_t)
                    # print(t_peak, initial)
                    x.append(t_peak - i - 1)
                    if initial_t > baseline and baseline_stop:
                        break
                    initial_t -= slop
                return x, line
        else:
            return [0, 0], [0, 0]

    except Exception as e:
        # print(f"Error: {e}")
        return [0, 0], [0, 0]
    finally:
        pass


# ==========Date: 11/07/2022====================

# explicit function to normalize array
def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr)) * diff) / diff_arr) + t_min
        norm_arr.append(temp)
    return np.array(norm_arr)


# Calculate all Q, S peak
def calculate_qs_peaks(x, peaks, margin=0, threshold=None):
    """Return all Q peak of the signal
    param x: signal, peaks: peak index, margin: margin between T-offset to P-Onset, threshold: threshold for Q peak
    return: [[q_peak_index], [q_peak_amplitude], [q_peak_amplitude_2nd]]
    """
    q_peak_index = []
    q_peak_amplitude = []
    s_peak_index = []
    s_peak_amplitude = []

    for ind in peaks:
        temp_q = left_deep(x, ind, margin=margin, lth=threshold)
        q_peak_index.append(temp_q[1])
        q_peak_amplitude.append(x[temp_q[1]])

        temp_s = right_deep(x, ind, margin=margin, lth=threshold)
        s_peak_index.append(temp_s[1])
        s_peak_amplitude.append(x[temp_s[1]])

    return np.array(q_peak_index), np.array(q_peak_amplitude), np.array(s_peak_index), np.array(s_peak_amplitude)


# Function For remove Smal Peaks from the signal
def remove_small_amplitute(out1, peaks, peak_prom, amp=0.05):
    """
    Param out1 -> array of elements, p -> peak index, peak_prom -> peak prominence, amp -> amplitude threshold
    Return: array of elements without small amplitude
    """
    p = [out1[i] for i in peaks[0]]
    for i in range(len(p)):
        st = peak_prom[1][i]
        et = peak_prom[2][i]
        lf = abs(p[i] - out1[peak_prom[1][i]])
        rf = abs(p[i] - out1[peak_prom[2][i]])
        #     print(lf, rf, st, et)
        if lf < amp and rf < amp:
            res = np.linspace(out1[st], out1[et], et - st)
            out1[st:et] = res
    return out1


# Rhythm Diagonostic
def rhythm_diagonostic(rhythm={}):
    """
    param: {'hr_rate': 86, 'hr_rate_type': 'Normal',
            'rr_type': 'Regular', 'qrs_dur': 118.0,
            'qrs_type': 'Narrow'}

    return: Diag result
    """
    hr_rate = rhythm['hr_rate']
    hr_type = rhythm['hr_rate_type']
    rr_type = rhythm['rr_type']
    qrs_type = rhythm['qrs_type']

    # if hr_type == 'Normal' and rr_type == 'Regular' and qrs_type == 'Narrow':
    #     # print("[::] Normal Sinus Rhythm")
    #     return "Normal Sinus Rhythm"

    # if rr_type == "Regular" and qrs_type == "Wide" and hr_rate > 100:
    #     # print("[::] AVNRT")
    #     return "AVNRT"

    # if rr_type == "Regular" and qrs_type == "Wide" and hr_rate < 100:
    #     # print("[::] VT: Ventricular Tachycardia")
    #     return "VT ventricular tachycardia"

    # if rr_type == "Irregular" and qrs_type == "Wide" and hr_rate < 150:
    #     # print("[::] VF: Ventricular Fibrillation")
    #     return "VF ventricular fibrillation"

    # if hr_rate > 150:
    #     # print("[::] Super Ventricular Tachycardia")
    #     return "Super Ventricular Tachycardia"


# Average polling
def average_polling(x, n=4, order=1):
    """Return average polling of the signal
    param x: signal, n: number of points
    return: average polling
    """
    for _ in range(order):
        avg_polling = []
        for i in range(0, len(x) - n, n):
            avg_polling.append(np.average(x[i:i + n]))
        x = avg_polling
    return np.array(avg_polling)


# Increase Number of  Points by concatenating
def increase_number_of_sample(x, sample=5000):
    """Return increased number of points of the signal
    param x: signal, n: number of points
    return: increased number of points
    """
    new_x = np.concatenate((x, x))
    while len(new_x) < sample:
        new_x = np.concatenate((new_x, x))
    return new_x[:sample]


# Dataframe concat
def concat_df(df):
    """
    param: dataframe
    return: concat daf
    """

    new_df = []
    for i in df.columns:
        # print("Concat: ", i)
        new_df = np.concatenate((new_df, df[i]))
    if len(new_df) >= 5000:
        return new_df
    else:
        return increase_number_of_sample(new_df)


def check_irregular(abc, threshold=20, cth=2):
    """

    :param abc: difference between peaks
    :param threshold: threshold value for comparision
    :param cth: number or false detection ignore
    :return: True / False
    """
    init = False
    count = 0
    for i in abc:
        if abs(i) > threshold:
            count += 1
            init = True
    if count >= cth: return init
    return False


def check_atrial_Rhythms(ATRIAL):
    # print("Atrial Rhythm Type: ", ATRIAL)
    if not ATRIAL["rhythm"] and not ATRIAL["pr_interval"] and not ATRIAL["pr_amp"]:
        # print("[::] Atrial Fibrillation")
        return "Atrial Fibrillation"
    else:
        return None


# Function to detect outlier in the given Dictionr set
def outlier_peaks_detector(bet, margin=1):
    """
    param: Dict
    return key list of outlier
    """
    # print(f"BET: {bet}")
    default = np.array([val for _, val in bet.items()])
    lb = np.mean(default) - margin * default.std()
    ub = np.mean(default) + margin * default.std()
    # print("Lower Limit: ", lb, "Upper Limit: ", ub)
    be_dif = []
    lbd = np.where(default > ub)[0]
    ubd = np.where(default < lb)[0]
    be_dif.extend(lbd)
    be_dif.extend(ubd)
    #     print(lbd, ubd)
    key = []
    for ind in be_dif:
        for i, v in enumerate(iter(bet)):
            if i == ind:
                key.append(v)
    return key


def validate_peaks(data, val=65):
    check = ["Lead_II", "Lead_I", "aVL", "aVR", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    data2 = []
    for i in check:
        a = len(data[i]["index"])
        # print(type(a), a)
        data2.append(a)
        # print(len(data[i]["index"]), data[i]["index"])

    length = Counter(data2).most_common(1)[0][0]
    for i in data2:
        if length == i:
            pass
        else:
            # perviou = data[check[data2.index(i) - 1]]["index"]
            perviou = data[check[np.where(data2 == i)[0][0] - 1]]["index"]
            # print(perviou)
            later = data[check[np.where(data2 == i)[0][0]]]["index"]
            # print(later)
            new_data = []
            new_val = []
            for j in perviou:
                for ind, k in enumerate(later):
                    if j + val > k > j - val:
                        new_data.append(k)
                        new_val.append(ind)

            data[check[np.where(data2 == i)[0][0]]]["index"] = new_data
            data[check[np.where(data2 == i)[0][0]]]["value"] = [data[check[np.where(data2 == i)[0][0]]]["value"][ii] for
                                                                ii in new_val]

    return data


# From List/Array
def remove_outlier(indices, margin=1.1):
    """
    param: list/array
    retun: numpy array
    """
    new_w = []
    lb, ub = np.mean(indices) - margin * np.std(indices), np.mean(indices) + margin * np.std(indices)
    for d in indices:
        if lb < d < ub:
            new_w.append(d)
    return np.array(new_w)


# Classes For Some Important Functions
class QRS2DProcessor:
    def __init__(self, qrs2d):
        self.qrs2d = qrs2d
        self.number_of_peaks = [len(i) for i in qrs2d]

    def max_occurrence(self):
        # Create a dictionary to store the count of each element
        counts = {}
        for num in self.number_of_peaks:
            if num in counts:
                counts[num] += 1
            else:
                counts[num] = 1

        # Find the element with the highest count
        max_count = 0
        max_num = None
        for num, count in counts.items():
            if count > max_count:
                max_count = count
                max_num = num

        return max_num

    def process_qrs2d(self, show=False):
        number_of_peak = self.max_occurrence()
        # outlier_leads = []
        for row in range(len(self.qrs2d)):
            if not len(self.qrs2d[row]) == number_of_peak:
                # outlier_leads.append(row)
                self.qrs2d[row] = [0] * number_of_peak

        # Transform the qrs2d list to a numpy array
        qrs2d = np.array(self.qrs2d)

        # Copy the original list to a new list
        lst = qrs2d.tolist()
        filled_lst = [row[:] for row in lst]

        # Loop through each column of the list
        for col in range(len(lst[0])):
            # Loop through each row of the column
            for row in range(len(lst)):
                # If the value is zero, find the nearest non-zero value in the same column
                if filled_lst[row][col] == 0:
                    # Find the nearest non-zero value above the current row
                    for i in range(row - 1, -1, -1):
                        if filled_lst[i][col] != 0:
                            filled_lst[row][col] = filled_lst[i][col]
                            break
                    # If there is no non-zero value above, find the nearest non-zero value below
                    if filled_lst[row][col] == 0:
                        for i in range(row + 1, len(lst)):
                            if filled_lst[i][col] != 0:
                                filled_lst[row][col] = filled_lst[i][col]
                                break

        # # print qrs2d in matrix form
        qrs2d = np.array(filled_lst)
        # if show:
        #     for i in range(len(qrs2d)):
        #         for j in range(len(qrs2d[i])):
        #             print(qrs2d[i][j], end=', ')
        #         print()
        return qrs2d


# ==========Date: 15/07/2022==================== Class for
# Peaks based Analysis of the signal
# ===================================================
# ==================================================
# ================================================
# ==============================================

class AnalysePeaks:
    def __init__(self, x, peaks, margin=0, threshold=None,  sampling_rate=70):
        """sampling Rate: 100 Hz Default"""
        self.x = x
        self.peaks = peaks
        self.margin = margin
        self.threshold = threshold
        self.peak_interval = []
        self.persamtim = 0
        self.sampling_rate =76
        self.heart_rate = 72
        self.rr_interval = 0
        self.rr_deviation = 0
        self.record_length = int(len(x) * (1 / sampling_rate))
      

    def cal_rr_interval(self):
        self.peak_interval = np.diff(self.peaks)
        self.abnormal = self.peak_interval[self.peak_interval > self.persamtim]

        # print("RR PEaks: ", self.peaks)
        self.avg_peak_inteval = np.average(self.peak_interval)
        self.rr_interval = self.avg_peak_inteval * (1000 / self.sampling_rate)
        # print("[::] R R Interval:\t --> {} ms".format(round(self.rr_interval)))
        return round(self.rr_interval)

    def get_rr_interval(self):
        return self.rr_interval
   
       
    def get_rr_deviation(self):
        self.rr_deviation = np.std(self.peak_interval) *(1000/self.sampling_rate)
        return round(self.rr_deviation) 
 
    def cal_heart_rate(self):
        if self.rr_interval == 0:
            raise ValueError("Please calculate R R Interval first: cal_rr_interval()")
        self.heart_rate = round(60 / (self.rr_interval / 1000))
        # print("[::] Heart Rate:\t --> {} bpm".format(self.heart_rate))
        return self.heart_rate

    # rr_interval = self.cal_rr_interval()
    # heart_rate = 60/(rr_interval/1000)

    def rate_analysis(self, show_out=True):
        """
        1. RATE:    (TOO FAST, TOO LOW, NORMAL)
            * 100 bpm > tachycardia
            * 60 bpm < bradycardia
            * 60 - 100 bpm normal
        """
        self.cal_rr_interval()
        self.cal_heart_rate()

        if show_out:
            if self.heart_rate > 100:
                # print("[::] Rythm:\t --> Tachycardia")
                return "Tachycardia"
            elif self.heart_rate < 60:
                # print("[::] Rythm:\t --> Bradycardia")
                return "Bradycardia"
            else:
                # print("[::] Rythm:\t --> Normal")
                return "Normal"
        else:
            if self.heart_rate > 100:
                return "Tachycardia"
            elif self.heart_rate < 60:
                return "Bradycardia"
            else:
                return "Normal"

    # Rythm analysis
    def rythm_analysis(self, threshold=20, show_out=True):
        """
        2. RHYTHM: (REGULAR or IRREGULAR)
            * Regular = R-R interval is constant
            * Irregular = R-R interval is not constant

        """
        sam_bw_peaks = abs(np.diff(np.diff(self.peaks)))
        # print(">>>>>>>>", np.diff(self.peaks))
        # print(">>>>>>>>", np.diff(np.diff(self.peaks)))
        # print(">>>>>>>> ", sam_bw_peaks)
        # print(">>>>>>>>", sam_bw_peaks.all() < threshold)
        if show_out:
            if check_irregular(sam_bw_peaks, threshold=20, cth=4):
                # print("[::] Rythm:\t --> Irregular")
                return "Irregular"
            else:
                # print("[::] Rythm:\t --> Regular")
                return "Regular"
        else:
            if check_irregular(sam_bw_peaks, threshold=20, cth=4):
                return "Irregular"
            else:
                return "Regular"

    # Get Beat Count
    def get_beat_count(self, show_out=True):
        if show_out:
            # print("[::] Record Length:\t --> {} seconds".format(self.record_length))
            # print("[::] Beat Count:\t --> {}".format(len(self.peaks)))
            return len(self.peaks)
        return len(self.peaks)


class QRSComplex:
    def __init__(self, x, qrs_peak, margin=0, sampling_rate=100, base_line=0):
        self.s_amp = None
        self.q_amp = None
        self.s_offset = None
        self.q_onset = None
        self.s_dur = None
        self.q_dur = None
        self.r_dur = None
        self.x = x
        self.qrs_peak = qrs_peak
        self.margin = margin
        self.result = []
        self.persamtim = 1 / sampling_rate
        self.base_line = base_line
        self.QRS_duration = 0
        self.Q_duration = 0
        self.R_duration = 0
        self.S_duration = 0
        self.qr_wave = []
        self.recall = False
        self.calculate_qrs_complex()

    def calculate_qrs_complex(self):
        """
        QRS Complex Calculate
        """
        self.result = []
        for qrs_i in self.qrs_peak:
            q_qrs = qrs_signal_finder(self.x, qrs_i, margin=self.margin)
            self.result.append(q_qrs)

        # print("\n###########################")
        # for item in self.result:
        #     print(colored(len(item[1]), 'green', 'on_red'), end=" ")  # type: ignore
        # print("\n###########################")

    def get_qrs_plot_data(self, show_out=True):
        """
        Return QRS plot data
        """
        # print(colored(len(self.result), "red"))  # type: ignore
        # print("***********************************")
        for item in self.result:
            print(colored(len(item[1]), "blue"), end=" ")  # type: ignore
        # print("***********************************")
        return self.result

    def get_qrs_duration(self, show_out=True):
        """
        Return QRS duration
        """
        duration = []
        for i in self.result:
            duration.append(len(i[0]) * self.persamtim)
        self.QRS_duration = np.floor(np.average(duration) * 1000)
        if self.QRS_duration > 180:
            self.margin = 0
            self.recall = True
            # print(colored("\n========\nMARGIN ERROR: Re-Calculating QRS duration...\n======\n", "red"))
            self.calculate_qrs_complex()
        # print("[::] QRS Duration:\t --> {} ms".format(self.QRS_duration))
        return duration, np.floor(self.QRS_duration)

    # get QRS Duration And Type of QRS
    def get_qrs_duration_type(self, show_out=True):
        """
        return: QRS Duration and type, (Narrow Wide of QRS)
        """
        duration, avg_duration = self.get_qrs_duration(show_out=False)
        if show_out:
            if avg_duration > 120:
                # print("[::] QRS Type:\t --> Wide")
                return "Wide"
            else:
                # print("[::] QRS Type:\t --> Narrow")
                return "Narrow"
        else:
            if avg_duration > 120:
                return "Wide"
            else:
                return "Narrow"

    # Get height/amplitude of QR, RS, S-offset amplitude
    def get_r_progression(self, q_point, s_point, show_out=False):
        """
        Parameter: q_point, s_point
        Return [QR, RS] amplitude
        """
        # list to Store QR, RS amplitude
        qr_prog = []
        rs_prog = []
        R_AMP = []
        Q_AMP = []
        S_AMP = []

        for q, r, s in zip(q_point, self.qrs_peak, s_point):
            if q is not None and r is not None and s is not None:
                # finding all qrs peak amplitude
                q_amp = self.x[q]
                r_amp = self.x[r]
                s_amp = self.x[s]

                # Calculate AMplitude difference between QR and RS
                qr_height = r_amp - q_amp
                sr_height = r_amp - s_amp
                qr_prog.append(qr_height)
                rs_prog.append(sr_height)
                R_AMP.append(r_amp)
                Q_AMP.append(q_amp)
                S_AMP.append(s_amp)

            else:
                continue

        if show_out:
            # print("[::] QR Amplitude:\t --> {} mv".format(np.average(qr_prog)))
            # print("[::] QR Max Amplitude:\t --> {} mv".format(np.max(qr_prog)))
            # print("[::] QR Min Amplitude:\t --> {} mv".format(np.min(qr_prog)))

            # print("[::] RS Amplitude:\t --> {} mv".format(np.average(rs_prog)))
            # print("[::] RS Max Amplitude:\t --> {} mv".format(np.max(rs_prog)))
            # print("[::] RS Min Amplitude:\t --> {} mv".format(np.min(rs_prog)))

            # print("[::] R Amplitude:\t --> {} mv".format(np.average(R_AMP)))
            # print("[::] R Max Amplitude:\t --> {} mv".format(np.max(R_AMP)))
            # print("[::] R Min Amplitude:\t --> {} mv".format(np.min(R_AMP)))

            # print("[::] Q Amplitude:\t --> {} mv".format(np.average(Q_AMP)))
            # print("[::] Q Max Amplitude:\t --> {} mv".format(np.max(Q_AMP)))
            # print("[::] Q Min Amplitude:\t --> {} mv".format(np.min(Q_AMP)))

            # print("[::] S Amplitude:\t --> {} mv".format(np.average(S_AMP)))
            # print("[::] S Max Amplitude:\t --> {} mv".format(np.max(S_AMP)))
            # print("[::] S Min Amplitude:\t --> {} mv".format(np.min(S_AMP)))

            return np.average(qr_prog), np.average(rs_prog)
        else:
            return np.average(qr_prog), np.average(rs_prog)

    # Find R Duration
    def get_r_duration(self, show_out=False):
        self.r_dur = []
        self.q_dur = []
        self.s_dur = []
        self.q_onset = []
        self.s_offset = []
        self.q_amp = []
        self.s_amp = []
        qrs = []
        # print("=============R Duration================")
        # plt.title("QRS Segments")
        for sig, ind in zip(self.result, self.qrs_peak):
            if len(sig[1]) > 1:
                # print(f"R Index: {ind}, len of sig: {len(sig[1])}")
                ind = sig[1].index(ind)
                # print(len(sig[0]), ind)
                # print("Line 980 index: ", ind)
                lp, l_indx = left_deep(sig[0], ind, lth=0.35)
                rp, r_indx = right_deep(sig[0], ind, lth=0.35)
                self.r_dur.append((rp + lp) * self.persamtim * 1000)
                self.q_dur.append(len(sig[0][:ind - lp]) * self.persamtim * 1000)
                self.s_dur.append(len(sig[0][ind + rp:]) * self.persamtim * 1000)
                self.q_onset.append(sig[1][0])
                self.s_offset.append(sig[1][-1])
                self.q_amp.append(sig[0][l_indx])
                self.s_amp.append(sig[0][r_indx])
                qrs.append(len(sig[0]) * self.persamtim * 1000)
                # print(f"Q index: {r_ind - l_indx}, Q val: {sig[0][l_indx]}")
            #     plt.plot(sig[0])
        # plt.show()

        self.r_amp = [self.x[i] for i in self.qrs_peak]

        # print("[::] R Duration: {} ms".format(self.r_dur))
        # print("[::] R Duration Average: {} ms".format(np.average(self.r_dur)))
        # print("[::] R Duration Max: {} ms".format(np.max(self.r_dur)))
        # print("[::] R Duration Min: {} ms".format(np.min(self.r_dur)))
        # print("[::] R Amplitude: {} mv".format(np.average(self.r_amp)))
        # print("=============Q Duratin=============")
        # print("[::] Q Duration: {} ms".format(q_dur))
        # print("[::] Q Duration Average: {} ms".format(np.average(self.q_dur)))
        # print("[::] Q Duration Max: {} ms".format(np.max(self.q_dur)))
        # print("[::] Q Duration Min: {} ms".format(np.min(self.q_dur)))
        # print("[::] Q Amplitude: {} mv".format(np.average(self.q_amp)))
        # print("=============S Duration=============")
        # print("[::] S Duration: {} ms".format(s_dur))
        # print("[::] S Duration Average: {} ms".format(np.average(self.s_dur)))
        # print("[::] S Duration Max: {} ms".format(np.max(self.s_dur)))
        # print("[::] S Duration Min: {} ms".format(np.min(self.s_dur)))
        # print("[::] S Amplitude: {} mv".format(np.average(self.s_amp)))
        # print("[::] QRS Duration: {} ms".format(np.average(qrs)))
        # print("====================================")
        # print("Q Onset: ", self.q_onset)
        # print("S Offset: ", self.s_offset)
        self.Q_duration = np.average(self.q_dur).round(3)
        self.R_duration = np.average(self.r_dur).round(3)
        self.S_duration = np.average(self.s_dur).round(3)
        return np.average(self.r_amp), np.average(self.q_amp), np.average(self.s_amp)

    def get_q_onset(self):
        return self.q_onset

    def get_s_offset(self):
        return self.s_offset

    def get_r_dur(self):
        return self.r_dur

    # Get QR Waveform
    def get_qr_waveform(self, show_out=False):
        for indx in self.qrs_peak:
            _, strart = left_deep(self.x, indx)
            wave = self.x[strart:indx]
            self.qr_wave.append(wave)
        if show_out:
            # print("[::] QR Waveform: {}".format(self.qr_wave))
            return self.qr_wave
        else:
            return self.qr_wave


class TWave:
    def __init__(self, x, qrs_peak, margin=0, sampling_rate=100,
                 conv_len=25, base_line=0, per_sample=0.0025):
        self.polarity = None
        self.left_prog = None
        self.t_x = x
        self.t_margin = margin
        self.t_qrs_peak = qrs_peak
        self.t_sampling_rate = sampling_rate
        self.t_conv_len = conv_len
        self.t_result = []
        self.s_peak_index = []
        self.center_point = find_center_point(self.t_x, self.t_qrs_peak)[0]
        self.t_slope = np.gradient(self.t_x)
        self.t_y = np.sin(np.linspace(0, np.pi, self.t_conv_len))
        self.t_amp_slop = np.convolve(self.t_slope, self.t_y, mode='same')
        self.t_base_line = base_line
        self.t_bthm = 0.5
        self.t_peak = []
        self.j_point = []
        self.per_sample = per_sample
        self.ST_type = "None"
        self.win_time = 150
        self.qrs_width = int((self.win_time / 1000) / self.per_sample)

    def get_t_wave(self, show=False, baseline=0, dur_mode=True, j_point=None):
        """
        Return: T wave signal, from s-offset to center point
        """
        # print("R-T Wave: {0:.3f}".format(self.center_point))
        self.left_prog = []
        self.t_result = []
        self.s_peak_index = []

        # ========== DATE: 09-10-2022 ===========

        # print("Line 1726 T: ", self.t_qrs_peak)
        t_start_point = [i + int(self.qrs_width / 2) for i in self.t_qrs_peak]
        # print("Line 1728 T: ", self.j_point)
        if j_point is None:
            self.j_point = j_point_finder(self.t_x, win_legth=4, qrs_peak=self.t_qrs_peak)
            # print("Line 1729 T: ", self.j_point)
        else:
            self.j_point = j_point
            # self.j_point = [[i, self.t_x[i]] for i in j_point]

        # print(f"JTM: {self.j_point}")
        if self.t_qrs_peak[0] > self.j_point[0]:
            # if self.t_qrs_peak[0] > self.j_point[0][0]:
            self.j_point.pop(0)

        # print("New R peak:\t", len(self.t_qrs_peak))
        # print("New J-Point:\t", len(self.j_point))
        # print("New C-Point:\t", len(self.center_point))

        # print("New R peak:\t", self.t_qrs_peak)
        # print("New J-Point:\t", self.j_point)
        # print("New C-Point:\t", self.center_point)
        self.polarity = []
        # for jp, cp in zip(self.j_point, self.center_point):
        for jp, cp in zip(t_start_point, self.center_point):
            # print("J-point: {}, C-point: {} ".format(jp, cp))
            self.t_result.append([self.t_x[jp:cp], jp])
            derivative = integration(np.diff(self.t_x[jp:cp]))  # np.convolve(np.diff(self.t_x[jp:cp]), np.ones(5))
            self.polarity.append(area_under_curve(derivative))
            # print("St Segments: ", list(self.t_x[jp:cp]))
            # nd2 = abs(np.diff(self.t_x[jp:cp], 2))
            # ind = np.argmax(abs(np.diff(self.t_x[jp:cp], 2)))
            # ==========

        # print("Polarity: ", self.polarity)
        if self.polarity.count(1) > self.polarity.count(0):
            self.ST_type = "positive"
            # print("Positive: ", self.polarity)
        elif self.polarity.count(1) < self.polarity.count(0):
            self.ST_type = "negative"
            # print("Negative: ", self.polarity)
        else:
            self.ST_type = "None"
            # print("Not Found: ", self.polarity)

        self.s_peak_index = self.j_point

        if show:
            plt.title("T wave Segments")
            for segment in [i[0] for i in self.t_result]:
                if len(segment) > 9:
                    # plt.plot(segment, color='red')
                    # y = np.cos(np.linspace(0, np.pi, 20))
                    y = np.ones(4)  # For moving Average
                    d = np.diff(segment)
                    # plt.plot(integration(d), color='green')
                    # print("T wave segment len: ", len(segment))
                    plt.plot(lowpass(segment, CUT_OFF_FREQUENCY=10), color='green')
            plt.axhline(0)
            plt.legend(["Original", "Integrated"])
            plt.show()

            plt.title("T wave average polling")
            for segment in [i[0] for i in self.t_result]:
                y = np.ones(10)  # For moving Average, we use linilear response for convolution. [10 -> window size]
                d = integration(np.diff(average_polling(segment, n=2, order=2)))
                # print("Nature: ", sum(d),"\t", area_under_curve(d))
                plt.plot(d, color='green')
            plt.axhline(0)
            plt.show()

        # print("Result: ", self.t_result)
        # for i in self.t_result:
        #     print(len(i))
        return self.t_result

    # TODO: correct t peak index
    def correct_t_peak_index(self):
        for ind in self.tp:
            # print("correct_t_peak_index: ", ind)
            pass

    def find_t_peak(self, show=False, baseline=0, polarity=1):
        """
        Parameters:
        - show: use to display graph if true
        - baseline: currently not in used,
        - polarity: most important, if 1 -> get the maximum value index,
                                    0 -> minimum value index,
                                    other -> for automatics detection based on the signal average value,
                                            it could be either positive or negative.

        Return [[peak, amp], [peak, amp], ...]
        """
        self.tp = []
        avg = []
        shift = 0
        self.s_wave = self.get_t_wave()
        for i in self.s_wave:
            if len(i[0]) > 0:
                # temp_peak = signal.find_peaks(i[0], prominence=0.001, distance=4)[0] + +i[1]
                # finding the maximum peak value for t peak
                # tem = i[0] - baseline

                # classification by area under the curve
                # temp = integration(np.diff(tem))
                if polarity == 1:
                    temp_peak = np.argmax(i[0]) + i[1]

                elif polarity == 0:
                    temp_peak = np.argmin(i[0]) + i[1]

                else:
                    # temp_peak = np.argmax(i[0]) + i[1] + shift
                    # cross-checking for negative p peak
                    av = np.average(i[0])
                    if abs(max(i[0]) - av) > abs(av - min(i[0])):
                        # print("POL:: Positive")
                        temp_peak = np.argmax(i[0]) + i[1] + shift
                        pass
                    else:
                        temp_peak = np.argmin(i[0]) + i[1] + shift
                        # print("POL:: Negative")
                        pass

                # print("S-Peak: ", temp_peak)
                self.tp.append([temp_peak, self.t_x[temp_peak]])
                avg.append(self.t_x[temp_peak])

        # print("[::] T-amp:\t --> {0:.3f} mv".format(np.average(avg)))
        # if np.average(avg) > 0.05:
        #     print("[::] Hypercute T wave: 1. Vasospasm, 2. Early Stemi")

        if show:
            plt.plot([i[1] for i in self.tp])
            plt.show()
        self.correct_t_peak_index()
        return self.tp, np.average(avg)

    def t_on_off(self, n_points=20):
        temp = []
        t_peaks = [i[0] for i in self.tp]
        for start, i in zip(t_peaks, self.s_wave):
            # print("ABC", start)
            ind, val = find_tangent_line(i[0],
                                         prev_n=n_points,
                                         t_peak=start, baseline_stop=True)
            # print("Offset: ", len(ind), ind[-1])
            temp.append([ind[-1]][0])
        # print("T Tangent OffSet: ", temp)

    def find_j_point(self, show=False, ld=None):
        """
        Return: J point and amplitude [[peak, amp], [peak, amp], ...]
        """
        self.j_point = []
        # for j_index in j_point_finder(self.t_x, win_legth=4, qrs_peak=self.t_qrs_peak, show=show):
        for j_index in self.j_point:
            if j_index:
                self.j_point.append([j_index, self.t_x[j_index]])

        # print("J-Point:\t --> {}".format(self.j_point))
        # print("QRS Peak: {}".format(self.t_qrs_peak))
        if self.t_qrs_peak[0] > self.j_point[0][0]:
            self.j_point.pop(0)
        if show:
            # print(f"J point: {self.j_point}")
            plt.title("J points")
            for items in self.j_point:
                plt.scatter(items[0], items[1], color='red')
            plt.plot(self.t_x)
            plt.show()
        return self.j_point

    # TODO: Find T duration
    def t_duration(self, t_index, lbsr=True, rbsr=True):
        """
        params: peaks, lbsr-> left baseline, rbsr-> right baseline
        Return : [t_duration, onset, offset]
        """
        if lbsr and rbsr:
            onset = t_index - left_deep(self.t_x, t_index, lth=self.t_base_line, slop=0.001)
            offset = t_index + right_deep(self.t_x, t_index, lth=self.t_base_line, slop=0.001)
            t = (offset[0] - onset[0]) * self.per_sample
            return t, onset[0], offset[0]

        elif lbsr and not rbsr:
            onset = t_index - left_deep(self.t_x, t_index, lth=self.t_base_line, slop=0.001)
            offset = t_index + right_deep(self.t_x, t_index, slop=0.001)
            t = t = (offset[0] - onset[0]) * self.per_sample
            return t, onset[0], offset[0]

        elif rbsr and not lbsr:
            onset = t_index - left_deep(self.t_x, t_index, slop=0.001)
            offset = t_index + right_deep(self.t_x, t_index, lth=self.t_base_line, slop=0.001)
            t = t = (offset[0] - onset[0]) * self.per_sample
            return t, onset[0], offset[0]

        else:
            onset = t_index - left_deep(self.t_x, t_index, slop=0.001)
            offset = t_index + right_deep(self.t_x, t_index, slop=0.001)
            t = (offset[0] - onset[0]) * self.per_sample
            return t, onset[0], offset[0]

    # TODO: Find T onset offset
    def t_onset_offset(self, show=False, error_const=10):
        """
        Return: P onset and offset
        """

        # Search for P-wave onset and offset index
        self.t_onset = []
        self.t_offset = []

        # Store P onset and offset Amplitute
        self.t_onset_val = []
        self.t_offset_val = []

        self.time_dur = []
        # print("[::] T-wave Amplitude:", [i[1] for i in self.tp])
        # plt.xlabel("Amplitude (mV)")
        # plt.ylabel("Contribution")
        # plt.hist([i[1]*10 for i in self.pp], bins=10)
        # plt.savefig("result/p_wave_amp.png")
        # temp = []
        t_peaks = [i[0] for i in self.tp]
        # print(f"Length of t peak: {len(t_peaks)}, s p: {len(self.s_wave)}")
        for start, i in zip(t_peaks, self.s_wave):
            if len(i[0]) > 0:
                stp = np.argmax(i[0]) + 1
                # print("ABC", start, "LEN:", len(i[0][stp:]))
                ind, val = find_tangent_line(i[0][stp:],
                                             prev_n=20,
                                             t_peak=start, baseline_stop=True)
                oti, otv = find_tangent_p_onset(i[0][:stp],
                                                prev_n=20,
                                                t_peak=start, baseline_stop=True)
                self.t_onset.append(oti[-1])
                self.t_onset_val.append(self.t_x[oti[-1]])
                # print("OTI: ", oti[-1])
                # print("Offset: ", ind[-1], val[-1])
                if ind[-1] < start:
                    self.t_offset.append(start + error_const)
                    self.t_offset_val.append(self.t_x[start + error_const])
                else:
                    self.t_offset.append(ind[-1])
                    # temp.append([ind[-1]][0])
                    self.t_offset_val.append(self.t_x[ind[-1]])

                if show and False:
                    plt.title("ON OFF")
                    plt.plot(i[0][:stp])
            else:
                # IF the length of the signal is 0 then assume default around the T peaks
                self.t_onset.append(start - error_const)
                self.t_onset_val.append(self.t_x[start - error_const])

                self.t_offset.append(start + error_const)
                self.t_offset_val.append(self.t_x[start + error_const])

        if False and show: plt.show()
        # print("T Tangent OffSet: ", temp)

        for i in self.tp:
            # onset = i[0] - left_deep(self.t_x, i[0], lth=self.t_base_line)
            # offset = i[0] + right_deep(self.t_x, i[0], lth=self.t_base_line)
            ti = self.t_duration(i[0], lbsr=True, rbsr=True)
            # print("---------->", ti)
            if ti[0] < 0.1:
                # print("First Trial")
                # self.t_onset.append(ti[1])
                # self.t_offset.append(ti[2])
                # self.t_onset_val.append(self.t_x[ti[1]])
                # self.t_offset_val.append(self.t_x[ti[2]])
                self.time_dur.append(ti[0])
            else:
                ti = self.t_duration(i[0], lbsr=True, rbsr=True)
                if ti[0] < 0.15:
                    # print("Second Trial")
                    # self.t_onset.append(ti[1])
                    # self.t_offset.append(ti[2])
                    # self.t_onset_val.append(self.t_x[ti[1]])
                    # self.t_offset_val.append(self.t_x[ti[2]])
                    self.time_dur.append(ti[0])
                else:
                    # print("Third Trial")
                    ti = self.t_duration(i[0], lbsr=True, rbsr=False)
                    # self.t_onset.append(ti[1])
                    # self.t_offset.append(ti[2])
                    # self.t_onset_val.append(self.t_x[ti[1]])
                    # self.t_offset_val.append(self.t_x[ti[2]])
                    self.time_dur.append(ti[0])

        # print("[::] T-onset:\t --> {0:.3f}".format(self.t_onset))
        # print("[::] T-offset:\t --> {0:.3f}".format(self.t_offset))

        if show:
            # print("[::] T-onset to P-offset:\t --> {0:.3f}".format(gap))
            print("[::] T-duration(ms):\t --> {0:.3f}".format(np.average(self.time_dur) * 1000))
            print("[::] T-mean duration(ms):\t --> {0:.3f}".format(np.mean(self.time_dur) * 1000))
            print("[::] T-std duration(ms):\t --> {0:.3f}".format(np.std(self.time_dur) * 1000))
            print("[::] T-max duration(ms):\t --> {0:.3f}".format(np.max(self.time_dur) * 1000))
            print("[::] T-min duration(ms):\t --> {0:.3f}".format(np.min(self.time_dur) * 1000))

            print("[::] T-onset Amplitute:\t --> {0:.3f} mv".format(np.average(self.t_onset_val)))
            print("[::] T-offset Amplitute:\t --> {0:.3f} mv".format(np.average(self.t_offset_val)))
            print("[::] T-Amplitute:\t --> {0:.3f} mv".format(
                np.average(self.t_onset_val) - np.average(self.t_offset_val)))
            print("[::] T-mean:\t --> {0:.3f} mv".format(np.mean(self.t_x)))
            print("[::] T-std:\t --> {0:.3f} mv".format(np.std(self.t_x)))
            print("[::] T-max:\t --> {0:.3f} mv".format(np.max(self.t_x)))
            print("[::] T-min:\t --> {0:.3f} mv".format(np.min(self.t_x)))

        return self.t_onset, self.t_offset, self.t_onset_val, self.t_offset_val, self.time_dur

    def get_t_onset(self):
        return self.t_onset

    def get_t_offset(self):
        return self.t_offset

    def get_time_dur(self):
        return self.time_dur

    # ST Wave Classification
    def st_wave_classification(self, show_out=False):
        # self.find_j_point()
        self.st_area = []
        # print("[::] J-Point:\t --> {}".format(self.j_point))
        # self.j_value = [i[1] for i in self.j_point]
        self.j_value = [self.t_x[i] for i in self.j_point]
        # print("[::] ", len(self.t_result), "value")
        for dt, j in zip(self.t_result, self.j_value):
            # print("######")
            y_up = np.array([i for i in dt[0] if i >= j])
            y_down = np.array([i for i in dt[0] if i < j])
            self.st_area.append([sum(abs(y_down - j)), sum(abs(y_up - j))])

        # print(self.st_area)
        # outcome = [i[0] - i[1] for i in self.st_area]
        # for i in outcome:
        #     if i > 0:
        #         print("down")
        #     elif i < 0:
        #         print("up")
        #     else:
        #         print("normal")

        return self.st_area


class PWave:
    def __init__(self, x, qrs_peak, margin=0, s_rate=100,
                 conv_len=25, base_line=0, per_sample=0.0025):
        self.left_prog = None
        self.time_dur = None
        self.pp = None
        self.p_offset = None
        self.avg = None
        self.p_wave = None
        self.p_onset = None
        self.x = x
        self.margin = margin
        self.qrs_peak = qrs_peak
        self.sampling_rate = s_rate
        self.conv_len = conv_len
        self.result = []
        # print("QRS_peak: ", self.qrs_peak)
        self.center_point_p = find_center_point(self.x, self.qrs_peak)[0]
        self.slope = np.gradient(self.x)
        self.y = np.sin(np.linspace(0, np.pi, self.conv_len))
        self.amp_slop = np.convolve(self.slope, self.y, mode='same')
        self.base_line = base_line
        self.bthm = 0.1
        self.p_peak = []
        self.per_sample = per_sample
        # print("P class init")

    def get_p_wave(self, show=False, shift_c=15):
        """
        Return: P wave signal, from center point to Q onset
        """
        self.result = []
        # print("Center point: {}".format(self.center_point_p))
        # print("R Peak: {}".format(self.qrs_peak))
        self.left_prog = []
        # print("Center point: {}".format(self.center_point_p))
        if self.center_point_p[0] > 0:
            self.center_point_p.insert(0, 0)
        if show: plt.title("P Wave Segments")
        # prev margin from R peak
        sta_ = np.int16(np.floor(0.2 * self.sampling_rate))
        # next margin from R peak
        _sta = np.int16(np.floor(0.07 * self.sampling_rate))

        # Time window Selector
        for ind, rlc in enumerate(self.qrs_peak):
            if rlc - sta_ > 0:
                a = rlc - sta_
                a = self.center_point_p[ind]
            else:
                a = 0
            b = rlc - _sta
            # print("Co-ordinate: ", a, b, rlc)
            self.result.append([self.x[a:b], a])
        if show:
            for sg in self.result:
                plt.plot(sg[0])
                plt.axhline(np.average(self.x[a:b]))
            plt.show()

        '''
        # Point Based Selector
        for rp, cp in zip(self.qrs_peak, self.center_point_p):
            # print("-->P-wave: ", cp, rp)
            p_end = left_deep(self.x, rp, slop=0)[0]
            lov = left_over(self.x, rp - p_end, lth=self.base_line, win=True, prst=self.per_sample)[0]
            p_offset = rp - p_end - lov
            # print("P-Offset: ", cp, "-->", p_offset)
            # print("NP-Offset: ", cp + shift_c, "-->", p_offset)
            # self.result.append([self.x[cp:p_offset], cp])

            # Checking polarity of the signal for peaks detection
            if p_offset - (cp + shift_c) > 20:
                self.result.append([self.x[cp + shift_c:p_offset], cp])
                seg = self.x[cp + shift_c:p_offset]
                # res = area_under_curve(average_polling(integration(np.diff(seg)), n=3, order=2))
            else:
                self.result.append([self.x[cp:p_offset], cp])
                seg = self.x[cp:p_offset]
            if show:
                plt.plot(seg)
                plt.axhline(np.average(seg))
                plt.show()
            '''
        return self.result

    def find_p_peaks(self, polarity=1, show=False):
        """
        param: polarity 1 -> maximum, 0 -> minimum
        Return [[peak, amp], [peak, amp], ...]
        """
        self.pp = []
        self.avg = []
        shift = 0
        self.p_wave = self.get_p_wave(shift_c=shift, show=False)
        ln = len(self.x)
        # print("Number of P wave detected: ", len(self.p_wave))
        for i in self.p_wave:
            # print(f"len of p wave: {len(i[0])}")
            if len(i[0]) > 0:
                if polarity == 1:
                    p_index = np.argmax(i[0]) + i[1] + shift

                elif polarity == 0:
                    p_index = np.argmin(i[0]) + i[1] + shift

                else:
                    p_index = np.argmax(i[0]) + i[1] + shift

                    # cross-checking for negative p peak
                    av = np.average(i[0])
                    if abs(max(i[0]) - av) > abs(av - min(i[0])):
                        # print("POL:: Positive")
                        pass
                    else:
                        p_index = np.argmin(i[0]) + i[1] + shift
                        # print("POL:: Negative")
                        pass

                # print("P-Peak: ", p_index)
                if p_index > ln - 1:
                    # print("Large P-Peak: ", p_index)
                    p_index = ln - 1

                self.pp.append([p_index, self.x[p_index]])
                self.avg.append(self.x[p_index])

        # print("[::] P-amp:\t --> {0:.3f} mv".format(np.average(self.avg)))
        # if np.average(self.avg) < 0.01:
        #     print("[::] P Wave Absent")

        return self.pp, np.average(self.avg), self.avg

    def p_wave_analysis(self, show=False):
        two_peak = []
        one_peak = []
        dia = "NA"

        y = np.linspace(1, 1, 5)
        for i in self.result:
            if len(i[0]) > 0:
                yy = np.convolve(i[0], y, 'same')
                th = yy.max() - (yy.max() / 3)
                lp = signal.find_peaks(yy, distance=4, height=th)

                if len(lp[0]) == 2 and not any(j > 10 for j in np.diff(lp[0])):
                    # print("Distance between: ", np.diff(lp[0]))
                    two_peak.append([i[0], lp])
                if len(lp[0]) == 1:
                    one_peak.append([i[0], lp])
                if show:
                    plt.title("P wave Analysis")
                    plt.axhline(th)
                    plt.plot(yy)
                    plt.scatter(lp[0], lp[1]["peak_heights"], color='red')

        if show: plt.show()

        # print("Number of Two Peaks: ", len(two_peak))
        # print("Number of One Peaks: ", len(one_peak))
        # print(f"average p wave amplitude: {np.average(self.avg):.4}")
        if len(two_peak) >= len(one_peak) or len(two_peak) >= 2:
            # print("There is M shape in P wave: ", len(two_peak))
            lam = []
            ram = []
            for i in two_peak:
                # print(i[1][1]["peak_heights"][0])
                if i[1][1]["peak_heights"][0] > i[1][1]["peak_heights"][1]:
                    lam.append(1)
                else:
                    ram.append(1)
            if len(lam) > len(ram):
                # print("Left Atrial Enlargements")
                dia = "Left Atrial Enlargements"
            elif len(lam) == len(ram):
                # print("LAE + RAE")
                dia = "LAE + RAE"
            else:
                # print("Right Atrial Enlargements")
                dia = "Right Atrial Enlargements"

        # height greater than 2.5 mm in inferior ECG leads (II, III, aVF); and greater than 1.5 mm in right sided
        # pre-cordial leads (V1, V2)
        elif np.average(self.avg) > 0.25:  # 1mm -> 0.1mv
            # print("Right Atrial Enlargements A")
            dia = "Right Atrial Enlargements A"
        elif abs(np.average(self.avg)) < 0.01:
            # print("No P wave Detected")
            dia = "No P Wave Detected"
        else:
            # print("NORMAL P wave: ", len(one_peak))
            dia = "NORMAL P wave"

        return dia

    def p_duration(self, p_index, lbsr=True, rbsr=True, pol=1):
        """
        params: peaks, lbsr-> left baseline, rbsr-> right baseline
        Return : [p_duration, onset, offset]
        """
        # if pol = 1 ie, p is positive, 0 - negetive
        if pol:
            if lbsr and rbsr:
                onset = p_index - left_deep(self.x, p_index, lth=self.base_line, slop=0.001)
                offset = p_index + right_deep(self.x, p_index, lth=self.base_line, slop=0.001)
                t = (offset[0] - onset[0]) * self.per_sample
                return t, onset[0], offset[0]

            elif lbsr and not rbsr:
                onset = p_index - left_deep(self.x, p_index, lth=self.base_line, slop=0.001)
                offset = p_index + right_deep(self.x, p_index, slop=0.001)
                t = t = (offset[0] - onset[0]) * self.per_sample
                return t, onset[0], offset[0]

            elif rbsr and not lbsr:
                onset = p_index - left_deep(self.x, p_index, slop=0.001)
                offset = p_index + right_deep(self.x, p_index, lth=self.base_line, slop=0.001)
                t = t = (offset[0] - onset[0]) * self.per_sample
                return t, onset[0], offset[0]

            else:
                onset = p_index - left_deep(self.x, p_index, slop=0.001)
                offset = p_index + right_deep(self.x, p_index, slop=0.001)
                t = (offset[0] - onset[0]) * self.per_sample
                return t, onset[0], offset[0]
        else:
            if lbsr and rbsr:
                onset = p_index - left_over(self.x, p_index, lth=self.base_line, slop=0.001)
                offset = p_index + right_over(self.x, p_index, lth=self.base_line, slop=0.001)
                t = (offset[0] - onset[0]) * self.per_sample
                return t, onset[0], offset[0]

            elif lbsr and not rbsr:
                onset = p_index - left_over(self.x, p_index, lth=self.base_line, slop=0.001)
                offset = p_index + right_over(self.x, p_index, slop=0.001)
                t = t = (offset[0] - onset[0]) * self.per_sample
                return t, onset[0], offset[0]

            elif rbsr and not lbsr:
                onset = p_index - left_over(self.x, p_index, slop=0.001)
                offset = p_index + right_over(self.x, p_index, lth=self.base_line, slop=0.001)
                t = t = (offset[0] - onset[0]) * self.per_sample
                return t, onset[0], offset[0]

            else:
                onset = p_index - left_over(self.x, p_index, slop=0.001)
                offset = p_index + right_over(self.x, p_index, slop=0.001)
                t = (offset[0] - onset[0]) * self.per_sample
                return t, onset[0], offset[0]

    def p_onset_offset(self):
        """
        Return: P onset and offset
        """

        # Search for P-wave onset and offset index
        self.p_onset = []
        self.p_offset = []

        # Store P onset and offset Amplitute
        self.p_onset_val = []
        self.p_offset_val = []

        self.time_dur = []
        # print("[::] P-wave Amplitude:", [i[1] for i in self.pp])
        # plt.xlabel("Amplitude (mV)")
        # plt.ylabel("Contribution")
        # plt.hist([i[1]*10 for i in self.pp], bins=10)
        # plt.savefig("result/p_wave_amp.png")

        # tangent method to Find p onset
        # print("P peaks:", len(self.pp), len(self.p_wave))
        temp_pp = [i[0] for i in self.pp]
        for sig, pe in zip(self.p_wave, temp_pp):
            # print("==>", sig[0])
            # tpl = np.argmax(sig[0])
            # print(tpl, pe)
            r = find_tangent_p_onset(sig[0], prev_n=1, baseline=0, t_peak=pe)[0]
            if r[-1] < 0:
                r[-1] = 0
            self.p_onset.append(r[-1])
            self.p_onset_val.append(self.x[r[-1]])
            # print("ABCD onset: ", r[-1])
            # print(r)
        # -----------------

        for i in self.pp:
            # onset = i[0] - left_deep(self.x, i[0], lth=self.base_line)
            # offset = i[0] + right_deep(self.x, i[0], lth=self.base_line)
            if i[1] >= 0:
                ti = self.p_duration(i[0], lbsr=True, rbsr=True, pol=1)  # ti -> Time Interval
                # print("->", ti)
                if ti[0] < 0.13:
                    # print("First Trial")
                    # self.p_onset.append(ti[1])
                    self.p_offset.append(ti[2])
                    # self.p_onset_val.append(self.x[ti[1]])
                    self.p_offset_val.append(self.x[ti[2]])
                    self.time_dur.append(ti[0])
                else:
                    ti = self.p_duration(i[0], lbsr=False, rbsr=True, pol=1)
                    if ti[0] < 0.13:
                        # print("Second Trial")
                        # self.p_onset.append(ti[1])
                        self.p_offset.append(ti[2])
                        # self.p_onset_val.append(self.x[ti[1]])
                        self.p_offset_val.append(self.x[ti[2]])
                        self.time_dur.append(ti[0])
                    else:
                        # print("Third Trial")
                        ti = self.p_duration(i[0], lbsr=False, rbsr=False, pol=1)
                        # self.p_onset.append(ti[1])
                        self.p_offset.append(ti[2])
                        # self.p_onset_val.append(self.x[ti[1]])
                        self.p_offset_val.append(self.x[ti[2]])
                        self.time_dur.append(ti[0])
            else:
                ti = self.p_duration(i[0], lbsr=True, rbsr=True, pol=0)  # ti -> Time Interval
                # print("->", ti)
                if ti[0] < 0.13:
                    # print("First Trial")
                    # self.p_onset.append(ti[1])
                    self.p_offset.append(ti[2])
                    # self.p_onset_val.append(self.x[ti[1]])
                    self.p_offset_val.append(self.x[ti[2]])
                    self.time_dur.append(ti[0])
                else:
                    ti = self.p_duration(i[0], lbsr=False, rbsr=True, pol=0)
                    if ti[0] < 0.13:
                        # print("Second Trial")
                        # self.p_onset.append(ti[1])
                        self.p_offset.append(ti[2])
                        # self.p_onset_val.append(self.x[ti[1]])
                        self.p_offset_val.append(self.x[ti[2]])
                        self.time_dur.append(ti[0])
                    else:
                        # print("Third Trial")
                        ti = self.p_duration(i[0], lbsr=False, rbsr=False, pol=0)
                        # self.p_onset.append(ti[1])
                        self.p_offset.append(ti[2])
                        # self.p_onset_val.append(self.x[ti[1]])
                        self.p_offset_val.append(self.x[ti[2]])
                        self.time_dur.append(ti[0])

        # print("[::] P-onset:\t --> {0:.3f}".format(self.p_onset))
        # print("[::] P-offset:\t --> {0:.3f}".format(self.p_offset))

        gap = [i[1] - i[0] for i in zip(self.p_onset, self.p_offset)]
        # print("[::] P-onset to P-offset:\t --> {0:.3f}".format(gap))
        # print("[::] P-Wave duration:\t --> {0:.3f} ms".format(np.average(self.time_dur) * 1000))
        # print("[::] P-onset Amplitude:\t --> {0:.3f} mv".format(np.average(self.p_onset_val)))
        # print("[::] P-offset Amplitude:\t --> {0:.3f} mv".format(np.average(self.p_offset_val)))
        # print("[::] P-Amplitude:\t --> {0:.3f} mv".format(np.average(self.p_onset_val) - np.average(self.p_offset_val)))
        return self.p_onset, self.p_offset, self.p_onset_val, self.p_offset_val, self.time_dur

    def get_p_onset(self):
        return self.p_onset

    def get_p_offset(self):
        # print("[::] P-offset:\t --> {0:.3f}".format(self.p_offset))
        return self.p_offset

    def get_time_dur(self):
        return self.time_dur
    
    # NEHA CODE 23/05/23
 
    def identify_SPCs(ecg_signal, sampling_rate):
        # Calculate the duration of each sample in milliseconds
        sample_duration = 1000 / sampling_rate
            
            # Set the threshold values for SPC criteria
        premature_threshold_ms = 120  # Maximum duration for premature beat in milliseconds
        p_wave_threshold_amplitude = 0.25  # Minimum amplitude difference for P wave
             
        spc_indices = []  # List to store the indices of SPCs
                
            # Iterate over the ECG signal to identify SPCs
        for i in range(1, len(ecg_signal) - 1):
            current_sample = ecg_signal[i]
            previous_sample = ecg_signal[i - 1]
            next_sample = ecg_signal[i + 1]
                    
                    # Check if the current sample is a premature beat
            if current_sample < previous_sample and current_sample < next_sample:
                
                # Check the duration of the premature beat
                duration_ms = sample_duration * (next_sample - previous_sample)
                if duration_ms < premature_threshold_ms:
                    
                    # Check the amplitude difference for P wave
                    p_wave_amplitude_diff = abs(ecg_signal[i - 1] - ecg_signal[i])
                    if p_wave_amplitude_diff > p_wave_threshold_amplitude:
                        spc_indices.append(i)
        
            return spc_indices

    # Function to get PR Types
    def pr_type(self, pr_interval):
        """
        Parameter: pr_interval list in milliseconds
        """
        # print(f"All PR Interval: {pr_interval}")
        # print(f"Adjecent Ration: {adjacent_ratios(np.diff(pr_interval).round(3))}")
        self.pri = np.average(pr_interval)
        # print("[::] ", abs(np.diff(pr_interval)))
        # if abs(np.diff(pr_interval)).any() < 20:
        #     # print("[::] PR nature: Constant")
        # else:
        #     # print("[::] PR nature: Variable")

        if 120 < self.pri < 200:
            # print("[::] PR Type: Normal")
            return "Normal PR"

        elif self.pri < 120:
            # print("[::] PR Type: Short")
            # print("[::] possible Pre-excitation syndromes")
            return "Short, possible Pre-excitation syndromes"

        elif 300 > self.pri > 200:
            # print("[::] PR Type: Prolonged")
            # print("[::] possible Prolonged PR Interval – AV block")
            return "Prolonged PR Interval, possible AV block"

        elif self.pri > 300:
            # print("[::] PR Type: Very Long")
            # print("[::] possible first degree heart block")
            return "Very Long, possible first degree heart block"


# Signal Classification
class SignalClassification:
    def __init__(self):
        self.dur = np.linspace(0, np.pi, 25)
        self.temp1 = np.sin(self.dur)
        self.temp2 = -self.temp1
        self.temp3 = np.sin(2 * self.dur)
        self.temp4 = -self.temp3
        self.temp5 = np.cos(self.dur)
        self.temp6 = -self.temp5
        # self.templates = [self.temp1, self.temp2, self.temp3, self.temp4, self.temp5, self.temp6]
        self.templates = [self.temp1]
        self.st_class = ["Positive"]
        # self.st_class = ["Positive", "T negetive", "P biphasic", "N biphasic", "P monophasic", "N monophasic"]

    def get_class(self, sg):
        self.result = []
        self.cresult = []
        for i in self.templates:
            cnv = np.convolve(sg, i, mode='same') / 10
            self.result.append(cnv)
            self.cresult.append(np.max(cnv))
            # print("[::] Max: ", np.max(cnv))
            # print("[::] Min: ", np.min(cnv))
            # print("[::] Mean: ", np.mean(cnv))
            # print("[::] Std: ", np.std(cnv))
            # print("----------------------------------")
        # ind = np.argmax(self.cresult)
        # print("[::] Class: ", self.cresult)
        # print("[::] ST- signal Class:\t ", self.st_class[ind])
        return self.result

    def get_class_2(self, sg):
        self.result = []
        self.cresult = []


# =======22-08-2022===========
class QtClassification:
    def __init__(self, qt_interval, qtc_interval=350, qt_sample=[], age=35, sex="male", sampling_rate=100):
        """
        # What is a normal QT interval?
        * .The length of a normal QT intervalTrusted Source varies by age and sex.
        * .
        * .For males and females below 15 years of age:
        * .
            * .Normal QT interval: 0.35 to 0.44 seconds
            * .Borderline QT interval: 0.44 to 0.46 seconds
            * .Prolonged QT interval: More than 0.46 seconds
        * .For adult males:
            * .
            * .Normal QT interval: 0.35 to 0.43 seconds
            * .Borderline QT interval: 0.43 to 0.45 seconds
            * .Prolonged QT interval: More than 0.45 seconds
        * .For adult females:
        * .
            * .Normal QT interval: 0.35 to 0.45 seconds
            * .Borderline QT interval: 0.45 to 0.47 seconds
            * .Prolonged QT interval: More than 0.47 seconds

        """
        self.result = {}
        self.final_result = {}
        self.angle = 0
        self.qt_interval = qt_interval
        self.qtc_interval = qtc_interval
        self.x = qt_sample
        self.age = age
        self.sex = sex
        self.sampling_rate = sampling_rate

    def get_result(self):
        # print("|| QT interval: ", self.qt_interval)
        if (self.sex == "male" or self.sex == "female") and self.age < 15:
            if 350 < self.qt_interval < 440:
                self.result["qt_type"] = "Normal Qt Interval"
            elif 440 <= self.qt_interval <= 460:
                self.result["qt_type"] = "Borderline Qt Interval"
            elif 320 <= self.qt_interval <= 350:
                self.result["qt_type"] = "Short Qt Interval"
            elif self.qt_interval <= 320:
                self.result["qt_type"] = "Very Short Qt Interval"
            else:
                self.result["qt_type"] = "Prolonged Qt Interval {0:.3f}".format(self.qt_interval)

        elif self.sex == "male" and self.age >= 15:
            if 350 < self.qt_interval < 430:
                self.result["qt_type"] = "Normal Qt interval"
            elif 430 <= self.qt_interval <= 450:
                self.result["qt_type"] = "Borderline Qt Interval"
            elif 320 <= self.qt_interval <= 350:
                self.result["qt_type"] = "Short Qt Interval"
            elif self.qt_interval <= 320:
                self.result["qt_type"] = "Very Short Qt Interval"
            else:
                self.result["qt_type"] = "Prolonged Qt Interval {0:.3f}".format(self.qt_interval)

        elif self.sex == "female" and self.age >= 15:
            if 370 < self.qt_interval < 450:
                self.result["qt_type"] = "Normal Qt interval"
            elif 450 <= self.qt_interval <= 470:
                self.result["qt_type"] = "Borderline Qt Interval"
            elif 340 <= self.qt_interval <= 370:
                self.result["qt_type"] = "Short Qt Interval"
            elif self.qt_interval <= 340:
                self.result["qt_type"] = "Very Short Qt Interval"
            else:
                self.result["qt_type"] = "Undefined Qt Interval {0:.3f}".format(self.qt_interval)

        else:
            self.result["qt_type"] = "Invalid Input"
            return False, self.result

        # print("[::] QT Type: ", self.result["qt_type"])
        return True, self.result


#neha code 25/04/2023
# =========================
# Ventricular activation time(VAT)



#  neha's code 12/11/2022
# =============================
# Arrhythmia_classification


if __name__ == "__main__":
    ar = [0, 0, 0, 1, 2, 3, 2, 1, 0, -1, -2, -3, -2, -1, 0, 1, 2, 3, 2, 1, 0, -1, -2, -3, -2, -1, 0, 1, 2, 3, 2, 1, 0,
          -1, -2, -3, -2, -1, 0, 1, 2, 3, 2, 1, 0, -1, -2, -3, -2, -1, 0, 0, 0]
    final = len(ar) - 1
    initial = 0
    print(cross_finder(ar, [initial], [final]))
    # https://ekg.academy/Atrial-Rhythms


