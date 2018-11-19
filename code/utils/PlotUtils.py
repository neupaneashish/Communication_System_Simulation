from matplotlib import pyplot as plt 
import numpy as np
from numpy import fft as ft 


def bode_plot(t, x_t, label = None, titleText = 'Frequency response'):
	N = np.size(t)
	if N <= 2:
		error('Time series signal must have more than 2 samples.')
		return None, None

	Fs = 1 / (t[1] - t[0])
	f = Fs/N * np.arange(np.floor(-(N - 1)/2) , np.floor((N - 1)/2) + 1)
	X_f = ft.fftshift(ft.fft(x_t))

	plt.subplot(2, 1, 1)
	plt.plot(f, 20 * np.log10(np.abs(X_f)), label = label)
	plt.title(titleText)
	plt.ylabel('|H(f)|, dB')

	plt.subplot(2, 1, 2)
	plt.plot(f, np.angle(X_f) / np.pi)
	plt.xlabel('f, Hz')
	plt.ylabel('< H(f), x pi rad/s')

	plt.show(block=False)
	plt.pause(0.0001)

	return f, X_f

def signal_plot(t, x_t, label = None, titleText = 'Signal', xText = 't, sec', yText = 'x(t)', discrete = False):
	if discrete:
		plt.stem(t, x_t, label = label)
	else:
		plt.plot(t, x_t, label = label)
	plt.title(titleText)
	plt.xlabel(xText)
	plt.ylabel(yText)
	plt.show(block=False)
	plt.pause(0.0001)

def eye_diagram_plot(t, x_t, T_p, titleText = 'Eye Diagram'):
	Fs = 1 / (t[1] - t[0])
	N = int(Fs * T_p)	# samples per symbol

	traces = [x_t[i:i+N] for i in range(0, np.size(x_t), N)] 
	t_trace = np.arange(0, T_p, 1/Fs)

	for trace in traces:
		plt.plot(t_trace, trace, linestyle = '-', color = 'blue')

	plt.title(titleText)
	plt.show(block=False)
	plt.xlabel('t')
	plt.pause(0.0001)


def save_current(filename):
	filename = '../../images/' + filename
	plt.savefig(filename)