from matplotlib import pyplot as plt 
import numpy as np
from numpy import fft as ft 


def bode_plot(t, x_t, label = None, titleText = 'Frequency response'):
	if np.size(t) <= 2:
		error('Time series signal must have more than 2 samples.')
		return None, None

	Fs = 1 / (t[1] - t[0])
	N = np.size(t) if np.size(t) > 1000 else 1000
	X_f = ft.fftshift(ft.fft(x_t, n = N))
	f = Fs/N * np.arange(np.floor(-(N - 1)/2) , np.floor((N - 1)/2) + 1)
	
	plt.subplot(2, 1, 1)
	plt.plot(f, 20 * np.log10(np.abs(X_f)), label = label)
	plt.title(titleText)
	plt.ylabel('|H(f)|, dB')

	plt.subplot(2, 1, 2)
	plt.plot(f, np.unwrap(np.angle(X_f)) / np.pi)
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

def eye_diagram_plot(t, x_t, T_p, eye_N = None, titleText = 'Eye Diagram'):
	Fs = 1 / (t[1] - t[0])
	N = int(T_p * Fs)
	if eye_N is None:
		eye_N = N	# samples per symbol

	traces = [x_t[i:i+eye_N+1] for i in range(0, np.size(x_t), N)]
	traces = traces[3:-3]
	t_trace = np.arange(0, (np.size(traces[0]))/Fs, 1/Fs)

	for trace in traces:
		if np.size(trace) < np.size(t_trace):
			continue
		plt.plot(t_trace, trace, linestyle = '-', color = 'blue')

	plt.title(titleText)
	plt.show(block=False)
	plt.xlabel('t')
	plt.pause(0.0001)

def show_image(img, titleText = 'Image'):
	plt.imshow(img, cmap='gray')
	plt.title(titleText)
	plt.show(block=False)
	plt.pause(0.0001)

def save_current(filename, out_type):
	if out_type == 'PLOT':
		path = '../../plots_out/'
	elif out_type == 'IMAGE':
		path = '../../images_out/'
	else:
		print('Invalid output save type: must be "PLOT" or "IMAGE"')		

	filename = path + filename
	plt.savefig(filename)