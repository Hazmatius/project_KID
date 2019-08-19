import os
import matplotlib.pyplot as plt
from scipy import misc
from skimage import io
import time

directory = '/Volumes/ALEX USB/project_KID/KID_memory/'
files = os.listdir(directory)
data = list()

time = list()
x_series = list()
y_series = list()
z_series = list()

for file in files:
    filename = file[0:-4]
    filename = filename.replace(' ', '')
    filename = filename.replace(']', '')
    filename = filename.split('[')
    timestamp = int(filename[0])
    command = filename[1]
    command = command.split(',')
    for i in range(len(command)):
        command[i] = float(command[i])
    data_pointer = (timestamp, command, directory + file)
    data.append(data_pointer)

    time.append(timestamp)
    x_series.append(command[0])
    y_series.append(command[1])
    z_series.append(command[2])

for x_val in x_series:
    print(x_val)


# plt.plot(x_series)
# plt.plot(y_series)
# plt.plot(z_series)
# plt.xlim([0, 500])
# plt.show()

for i in range(20):
    img = io.imread(data[0][2])
    plt.imshow(img)
    plt.show()
    time.sleep(1)
