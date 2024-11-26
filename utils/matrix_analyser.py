from utils import function as func
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.max_open_warning"] = False


def average_pooling(matrix, pool_size=2):
    # Get the dimensions of the matrix
    rows, cols = len(matrix), len(matrix[0])

    # Calculate the new dimensions of the matrix after pooling
    new_rows = rows // pool_size
    new_cols = cols // pool_size

    # Create a new matrix with the new dimensions
    new_matrix = [[0 for _ in range(new_cols)] for _ in range(new_rows)]

    # Iterate over the elements in the new matrix
    for i in range(new_rows):
        for j in range(new_cols):
            # Get the values for the region of the original matrix
            # that the current element in the new matrix represents
            region = [matrix[i * pool_size + k][j * pool_size + l] for k in range(pool_size) for l in range(pool_size)]

            # Calculate the average value for the region
            avg_value = sum(region) / len(region)

            # Set the value in the new matrix
            new_matrix[i][j] = avg_value

    # Return the new matrix
    return np.array(new_matrix)


data = np.load(r"C:\Users\Nitish\PycharmProjects\ECG_V4\vertical.npy", allow_pickle=True)
print(data.shape)
for led in range(len(data)):
    temp = []
    plt.figure()
    for i in range(len(data[led])):
        am = np.array(func.lowpass(data[led][i], CUT_OFF_FREQUENCY=10))**2
        am = func.average_polling(am)
        # am[am < 0] = 0
        # am[am > 0] = 250
        plt.plot(5 + am * 5)
        temp.append(am)
        # plt.step(range(len(data[led][i])), data[led][i])
    temp = np.array(temp)
    print("Max Value: ", np.mean(temp))
    temp[temp < np.mean(temp)] = 0
    temp[temp >= np.mean(temp)] = 1
    temp = np.dot(temp.T, temp)
    plt.imshow(temp, cmap='gray')
    # break
plt.show()
plt.close()

# ecg_sig = np.load("C:/Users/Admin/PycharmProjects/CT-PR-0038-V2/ecg_signal.npy", allow_pickle=True) ** 2
# # plt.plot(20 + (ecg_sig*100).T)
# ecg_sig = np.int16(np.dot(ecg_sig.T, ecg_sig) * 100)
# # ecg_sig = average_pooling(ecg_sig, pool_size=4)
# ecg_sig[ecg_sig < 0] = 0
# print(ecg_sig)
# plt.imshow(ecg_sig, cmap='gray')
# plt.show()
