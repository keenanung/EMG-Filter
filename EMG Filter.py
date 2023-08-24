import csv
import math
from scipy.fft import rfft, rfftfreq, irfft
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

EMG_data = "EMG_Datasets.csv"

with open(EMG_data, mode='r', newline='') as file:
    csv_reader = csv.reader(file)
    relaxed = []
    contracted = []
    time = []
    next(csv_reader)

# Create the relaxed and contracted arrays to be fed to fft
    for row in csv_reader:      
      relaxed = relaxed + [float(row[1])]
      contracted = contracted + [float(row[2])]
      time = time + [float(row[0])]

# Create the function to compute RMS:

def computeRMS(arrayOfValues):
    a = 0
    for value in arrayOfValues:
        a = a + (value**2)
    RMS = 1/(len(arrayOfValues)-1) * math.sqrt(a)
    return RMS


# Plot the signals in the time domain before the filters

plt.plot(time, contracted )
plt.xlabel('Time (seconds)')
plt.ylabel('Magnitude')
plt.title('Contracted Original Signal')
plt.show()

plt.plot(time, relaxed )
plt.xlabel('Time (seconds)')
plt.ylabel('Magnitude')
plt.title('Relaxed Original Signal')
plt.show()

# Compute the RMS of the Relaxed Signal

relaxedRMS = computeRMS(relaxed)
print("Relaxed RMS:" + str(relaxedRMS))

# Compute the RMS of the Contracted Signal

contractedRMS = computeRMS(contracted)
print("Contracted RMS:" + str(contractedRMS))


# Plot the signals in the frequency domain before the filters

sampleRate = 0.000488328938372888 #time between each sample in the csv (in seconds)

cy = rfft(contracted)
xf = rfftfreq(len(contracted), sampleRate)
xw = []

for value in xf:
    value = value * 2 * 3.141592654
    xw = xw + [value]

plt.plot(xw, np.abs(cy))
plt.xlabel('Frequency [radians / second]')
plt.ylabel('Magnitude')
plt.title('Contracted Original Signal FFT')
plt.show()

ry = rfft(relaxed)
plt.plot(xw, np.abs(ry))
plt.xlabel('Frequency [radians / second]')
plt.ylabel('Magnitude')
plt.title('Relaxed Original Signal FFT')
plt.show()

# Construct the band stop filter. Prefix s before variables indicates 'stop'

sb, sa = signal.butter(2, [361.2831552,392.6990817], 'bandstop', analog=True)
sw, sh = signal.freqs(sb, sa, xw) # sw is frequency, sh is gain. Calculating at all frequencies in rxw (fourier transform)

# Construct the band pass filter. Prefix p before variables indicates 'pass'

pb, pa = signal.butter(2, [0.6283185307,2827.433388], 'bandpass', analog=True)
pw, ph = signal.freqs(pb, pa, xw) # pw is frequency, ph is gain. Calculating at all frequencies in rxw (fourier transform)

# Filter the Signals with the band stop

stopFilteredRY = []
stopFilteredCY = []
fullyFilteredRY = []
fullyFilteredCY = []

n = 0
for value in cy:
    value = value * sh[n] 
    stopFilteredCY = stopFilteredCY + [value]
    n = n +1

n = 0
for value in ry:
    value = value * sh[n] 
    stopFilteredRY = stopFilteredRY + [value]
    n = n + 1

# Filter the Signals with the band pass

n = 0 
for value in stopFilteredCY:
    value = value * ph[n] 
    fullyFilteredCY = fullyFilteredCY + [value]
    n = n +1

n = 0 
for value in stopFilteredRY:
    value = value * ph[n] 
    fullyFilteredRY = fullyFilteredRY + [value]
    n = n +1

# Plot the Filtered Signals In the Frequency Domain

plt.plot(xw, np.abs(fullyFilteredCY))
plt.xlabel('Frequency [radians / second]')
plt.ylabel('Magnitude')
plt.title('Contracted Filtered Signal FFT')
plt.show()

plt.plot(xw, np.abs(fullyFilteredRY))
plt.xlabel('Frequency [radians / second]')
plt.ylabel('Magnitude')
plt.title('Relaxed Filtered Signal FFT')
plt.show()

# Plot the Signals Back In the Time Domain. Note use xf rather than xw since scipy requires frequency in Hz

filteredRelaxed = irfft(fullyFilteredRY)
filteredContracted = irfft(fullyFilteredCY)

plt.plot(time, filteredContracted)
plt.xlabel('Time (seconds)')
plt.ylabel('Magnitude')
plt.title('Contracted Filtered Signal')
plt.show()

plt.plot(time, filteredRelaxed )
plt.xlabel('Time (seconds)')
plt.ylabel('Magnitude')
plt.title('Relaxed Filtered Signal')
plt.show()

# Find the RMS of the Filtered Signals in the Time Domain. Print This Value

filteredRelaxedRMS = computeRMS(filteredRelaxed)
print("Relaxed Filtered RMS:" + str(filteredRelaxedRMS))

filteredContractedRMS = computeRMS(filteredContracted)
print("Contracted RMS:" + str(filteredContractedRMS))

# Export CSV Files of both filtered relaxed and filtered contracted files



