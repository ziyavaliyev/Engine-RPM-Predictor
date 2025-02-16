import json
import numpy as np
import matplotlib.pyplot as plt
import logging
from utils.utils import volt2pressure, fast_level, pressure2dBSPL, save_plot, create_dir, plot_gps_data
from scipy.signal import butter, sosfilt, welch, find_peaks
import torch
import pandas as pd

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("lens.log"),
                              logging.StreamHandler()])
logging.info("Start logging")

file_path = 'data/240304_120336.json' #'240213_130125_reduziert.json'

with open(file_path, 'r') as file:
    data = json.load(file)

# Filter the microphone data
micdata = [md for md in data if list(md.keys())[1] == 'MD']

# Create a mapbased on the given GPS data
gpsdata = [gps for gps in data if list(gps.keys())[1] == 'Y']

logging.info("Creating a map...")
plot_gps_data(gpsdata)
logging.info("The map is saved in the maps folder")

#Set the start to zero
first_entry = micdata[0]['TS']
for i in range(len(micdata)):
    micdata[i]['TS'] = micdata[i]['TS'] - first_entry

#Find the maximum value between all list entries in each MD dictionary key
max_value = max(abs(value) for dictionary in micdata for value in dictionary['MD'])

#Interpolate the MD values and directly normalize
interpolated_dict = []
for i in range(len(micdata)-1):
    for j in range(len(micdata[i-1]["MD"])):
        delta = (micdata[i+1]["TS"] - micdata[i]["TS"]) / (len(micdata[i]["MD"]))
        interpolated_dict.append({'TS':micdata[i]["TS"]+j*delta, 'MD':micdata[i]["MD"][j]/max_value})
    if (i + 1) % ((len(micdata)-1) // 10) == 0:
        logging.info(f"Interpolating MD values: {((i + 1) / (len(micdata)-1)) * 100:.0f}%")

#Handle the last timestamp -> extrapolate
delta = (micdata[-1]["TS"] - micdata[-2]["TS"])/len(micdata[-1]["MD"])
for i in range(len(micdata[-1]["MD"])):
    interpolated_dict.append({'TS':micdata[-1]["TS"]+i*delta, 'MD':micdata[-1]["MD"][i]/max_value})

logging.info("Interpolation is done")

# Create the directories
ts_dir = create_dir()

#Pack TS and MD data to separate lists, once for the whole data and once for smaller chunk
TS_list = [entry['TS']/1000000 for entry in interpolated_dict]
MD_list = [entry['MD'] for entry in interpolated_dict]
plt.plot(MD_list)
save_plot("1_normalized", ts_dir)


#TS_list_sec = [np.ceil(entry['TS']/1000) for entry in interpolated_dict[::20]]
#MD_list_sec = [entry['MDö'] for entry in interpolated_dict[::20]]
"""
db = volt2pressure(MD_list, 0, sensitivity=-27.706)
plt.plot(TS_list, db)
save_plot("2_volt2pressure", ts_dir)
"""

filtered_MD_list = sosfilt(butter(6, 1/1200, btype='high', analog=False, output='sos'), MD_list)
plt.plot(filtered_MD_list)
save_plot("2_filtered", ts_dir)

db = volt2pressure(filtered_MD_list, 0, sensitivity=-27.706)
plt.plot(db)
save_plot("3_volt2pressure", ts_dir)

weighted_time_freq = fast_level(filtered_MD_list, 22050)[1]
plt.plot(weighted_time_freq)
save_plot("4_weighted_time_freq", ts_dir)

weighted_time_freq_dbspl = pressure2dBSPL(weighted_time_freq) #dbspl
plt.plot(weighted_time_freq_dbspl)
save_plot("4a_pressure2dBSPL", ts_dir)

df = pd.read_csv("data/data.csv", encoding='ISO-8859-1')
temp = np.array(df["PID 0C  E8"])
x = list(map(int, (temp[830:830+len(weighted_time_freq_dbspl)])))

data = {
    'x': x,
    'weighted_time_freq_dbspl': weighted_time_freq_dbspl
}
df = pd.DataFrame(data)
file_path = './results/dbspl_to_rpm.xlsx'
df.to_excel(file_path, index=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot x over its indices on the first subplot
ax1.plot(x, marker='o', linestyle='-', color='b')
ax1.set_title('Plot of x over its indices')
ax1.set_xlabel('Index')
ax1.set_ylabel('x')

# Plot weighted_time_freq_dbspl over its indices on the second subplot
ax2.plot(weighted_time_freq_dbspl, marker='s', linestyle='--', color='r')
ax2.set_title('Plot of weighted_time_freq_dbspl over its indices')
ax2.set_xlabel('Index')
ax2.set_ylabel('weighted_time_freq_dbspl')

# Display the plots
plt.tight_layout()
plt.show()

"""
### Welch function, find peaks, fundamental frequencies and rpm
welch_MD = welch(filtered_MD_list) #Welch
plt.plot(welch_MD[1])
save_plot("5_welch", ts_dir)

fft_md = np.abs(np.fft.fft(filtered_MD_list))# fft(filtered_MD_list) -> peaks, drehzahl -> plot
plt.plot(fft_md[1000000:6000000])
save_plot("6_fft_md", ts_dir)

peaks = find_peaks(fft_md)[0] #welch_MD[1])[0]
grundfrequenzen_ps = [fft_md[i] for i in peaks]
#grundfrequenzen_sf = [welch_MD[0][i] for i in peaks]

drehzahlen = [gf * 60 for gf in grundfrequenzen_ps]
plt.plot(drehzahlen[500000:2000000])
save_plot("7_drehzahlen", ts_dir)


### Root Mean Square Calculation
rms_weighted_time_freq = np.sqrt(np.mean(np.square(weighted_time_freq)))
logging.info(f"Root Mean Square (RMS) of weighted_time_freq: {rms_weighted_time_freq}")

# Create a figure and a set of subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: TS_list vs MD_list
axs[0].plot(TS_list, MD_list)
axs[0].set_title('After normalization')
axs[0].set_xlabel('TS_list')
axs[0].set_ylabel('MD_list')

# Plot 2: TS_list vs db
axs[1].plot(TS_list, db)
axs[1].set_title('After volt2pressure')
axs[1].set_xlabel('TS_list')
axs[1].set_ylabel('db')

# Plot 3: TS_list vs filtered_MD_list
axs[2].plot(TS_list, filtered_MD_list)
axs[2].set_title('After butter and sosfilt')
axs[2].set_xlabel('TS_list')
axs[2].set_ylabel('filtered_MD_list')

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.savefig('my_plot.png')
plt.show()

print("Done")"""