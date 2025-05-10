# Flat-Top Sampling
**Name:** Ranjith Ganesh B.  
**Roll Number:** 212223060222

## Aim
Write a Python program for the **Construction and Reconstruction of Flat-Top Sampling**.

---

## Tools Required
1. **Python**: A versatile programming language used for scientific computing and signal processing.  
2. **NumPy**: A powerful numerical library in Python for performing array-based operations and mathematical computations.  
3. **Matplotlib**: A plotting library for generating high-quality graphs and visualizations of data, essential for demonstrating the sampling process.  

---

## Program

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

fs = 1000  # Sampling frequency (samples per second)
T = 1      # Duration in seconds
t = np.arange(0, T, 1/fs)  # Time vector
fm = 5     # Frequency of message signal (Hz)

message_signal = np.sin(2 * np.pi * fm * t)

pulse_rate = 50  # pulses per second
pulse_train_indices = np.arange(0, len(t), int(fs / pulse_rate))
pulse_train = np.zeros_like(t)
pulse_train[pulse_train_indices] = 1

flat_top_signal = np.zeros_like(t)
sample_times = t[pulse_train_indices]
pulse_width_samples = int(fs / (2 * pulse_rate)) # Adjust pulse width as needed

for i, sample_time in enumerate(sample_times):
    index = np.argmin(np.abs(t - sample_time))
    if index < len(message_signal):
        sample_value = message_signal[index]
        start_index = index
        end_index = min(index + pulse_width_samples, len(t))
        flat_top_signal[start_index:end_index] = sample_value

def lowpass_filter(signal, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, signal)

cutoff_freq = 2 * fm  # Nyquist rate or slightly higher
reconstructed_signal = lowpass_filter(flat_top_signal, cutoff_freq, fs)

plt.figure(figsize=(14, 10))

plt.subplot(4, 1, 1)
plt.plot(t, message_signal, label='Original Message Signal')
plt.title('Original Message Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 2)
plt.stem(t[pulse_train_indices], pulse_train[pulse_train_indices], basefmt=" ", label='Ideal Sampling Instances')
plt.title('Ideal Sampling Instances')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(t, flat_top_signal, label='Flat-Top Sampled Signal')
plt.title('Flat-Top Sampled Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(t, reconstructed_signal, label=f'Reconstructed Signal (Low-pass Filter, Cutoff={cutoff_freq} Hz)', color='green')
plt.title('Reconstructed Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```
## Output Waveform

###  Simulation Output:
*The waveform plot displayed from the Python program showing:*
- Original Message Signal  
- Ideal Sampling Instances  
- Flat-Top Sampled Signal  
- Reconstructed Signal  

### Hardware Experiment Output Waveform:
**Image:**  
![image](https://github.com/user-attachments/assets/1c7d0912-fea9-420b-b07a-c2f7a2504e3b)
![WhatsApp Image 2025-05-10 at 16 07 52_2f423827](https://github.com/user-attachments/assets/5cde5f49-50ae-453b-a701-997799c67aa8)

---

## Results

Flat Top Sampling was successfully simulated using a pulse train to sample a continuous-time message signal.  
- The **sampled waveform** retained the **amplitude** of the original signal at discrete intervals with **flat tops**.  
- The **reconstructed signal**, obtained using a **low-pass filter**, closely resembles the original signal.  
- The **hardware output waveform** confirms accurate **flat-top representation** of the message signal samples.

## Conclusion:
The experiment demonstrates both theoretical and practical aspects of flat-top sampling and reconstruction effectively.
