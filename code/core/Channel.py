import numpy as np
from scipy import signal as sp 
from matplotlib import pyplot as plt
import PlotUtils as myplt

class Channel():

	def __init__(self, b, a, Fs, T_p, name):
		N = int(Fs * T_p)
		self.__b = np.array([[b[i], *list(np.zeros(N - 1))] for i in range(np.size(b))]).flatten()
		self.__a = np.array([[a[i], *list(np.zeros(N - 1))] for i in range(np.size(a))]).flatten()
		self.__T_p = T_p
		self.__Fs = Fs
		self.__type = name

	@property
	def name(self):
		return self.__type
	

	def transmit_signal(self, signal):
		filtered_signal = sp.lfilter(self.__b, self.__a, signal)
		#filtered_signal = filtered_signal[:np.size(signal)]
		t = np.arange(0, np.size(filtered_signal)/self.__Fs, 1/self.__Fs) 
		return t, filtered_signal

	def plot_impulse_response(self, titleText = None):
		if titleText is None:
			titleText = 'Impulse Response ' + self.__type + ' Channel'
		t = np.arange(0, np.size(self.__b)/self.__Fs, 1/self.__Fs)
		myplt.signal_plot(t, self.__b, yText = 'h(t)', titleText = titleText, discrete = True)

	def plot_freq_response(self, titleText = None):
		if titleText is None:
			titleText = 'Frequency Response ' + self.__type + ' Channel'

		w, H_f = sp.freqz(self.__b, self.__a)
		f = w / (2 * np.pi) * self.__Fs
		plt.subplot(2,1,1)
		plt.plot(f, np.abs(H_f))
		plt.title(titleText)
		plt.ylabel('|H(f)|')

		plt.subplot(2,1,2)
		plt.plot(f, np.angle(H_f) / np.pi)
		plt.ylabel('<H(f), x pi rad/s')
		plt.xlabel('f, Hz')
		
		plt.show(block=False)
		plt.pause(0.0001)

	def plot_eye_diagram(self, transmitter, num_symbols = 1000):
		random_symbols = np.random.randint(transmitter.M, size = num_symbols)
		titleText = 'Eye Diagram, Transmitter : ' + transmitter.name + ' , Channel : ' + self.name
		t, modulated_signal = transmitter.transmit_symbols(random_symbols)
		t, transmitted_signal = self.transmit_signal(modulated_signal)
		myplt.eye_diagram_plot(t, transmitted_signal, self.__T_p, titleText = titleText)

	def add_awgn(self, signal, mean = 0, var = 1):
		noise = mean + np.random.randn(np.size(signal)) * np.sqrt(var)
		return signal + noise

	def plot_eye_diagram_after_noise(self, transmitter, mean = 0, var = 1, num_symbols = 1000):
		random_symbols = np.random.randint(transmitter.M, size = num_symbols)
		titleText = ('Eye Diagram, TX : ' + transmitter.name + ' , CH : ' + self.name + 
						' , Noise Var: %.4f' % var)
		t, modulated_signal = transmitter.transmit_symbols(random_symbols)
		t, transmitted_signal = self.transmit_signal(modulated_signal)
		transmitted_signal = self.add_awgn(transmitted_signal, mean = mean, var = var)
		myplt.eye_diagram_plot(t, transmitted_signal, self.__T_p, titleText = titleText)


class OutdoorChannel(Channel):
	def __init__(self, Fs, T_p):
		h_outdoor = np.array([0.5, 1, 0, 0.63, *list(np.zeros(4)), 0.25, 
							*list(np.zeros(3)), 0.16, *list(np.zeros(12)), 0.1])
		super().__init__(h_outdoor, np.ones(1), Fs, T_p, 'Outdoor')

class IndoorChannel(Channel):
	def __init__(self, Fs, T_p):
		h_indoor = np.array([1, 0.4365, 0.1905, 0.0832, 0, 0.0158, 0, 0.003])
		super().__init__(h_indoor, np.ones(1), Fs, T_p, 'Indoor')

if __name__ == "__main__":
	print('This is the class definition - run testChannel to test it!')
