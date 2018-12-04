import sys
sys.path.append('../utils')

import numpy as np
from scipy import signal as sp 
import PlotUtils as myplt
from constants import PAMCoeffs as coeffs


class Receiver():
	def __init__(self, h_t, t, T_pulse, Fs, Ts_optimal, name = 'rx'):
		self.__h_t = h_t
		self.__t = t
		self.__T_p = T_pulse
		self.__Ts_optimal = Ts_optimal
		self.__Fs = Fs
		self.__type = name

	@property
	def name(self):
		return self.__type

	def plot_impulse_response(self, titleText = None):
		if titleText is None:
			titleText = 'Impulse Response ' + self.__type + ' Receiver'
		myplt.signal_plot(self.__t, self.__h_t, yText = 'h(t)', titleText = titleText)


	def plot_freq_response(self, titleText = None):
		if titleText is None:
			titleText = 'Frequency Response ' + self.__type + ' Receiver'
		myplt.bode_plot(self.__t, self.__h_t, titleText = titleText)

	def receive_signal(self, signal):
		rec_signal = np.convolve(signal, self.__h_t, mode = 'full')
		
		# donot truncate the delayed part just negative time - truncation code commented out
		start = np.argwhere(np.abs(self.__t) <= 1/(2 *self.__Fs)).flatten()[0]
		#rec_signal = rec_signal[start : start + np.size(signal) + np.size(self.__T_p)]
		rec_signal = rec_signal[start:]
		#pos_time0 = np.argwhere(np.abs(self.__t) <= 1/(2 *self.__Fs)).flatten()[0]
		pos_time0 = 0

		t = np.arange(-pos_time0/self.__Fs,(np.size(rec_signal)-pos_time0)/self.__Fs, 1/self.__Fs)
			
		return t, rec_signal

	def sample_sig_to_bits(self, t, signal, threshold = 0):
		Ns_optimal = int(self.__Ts_optimal * self.__Fs)

		sampled_bits = np.array([1 if signal[i] > threshold else 0 for i in range(Ns_optimal, np.size(signal), Ns_optimal)])
		return sampled_bits

	def plot_eye_diagram(self, transmitter, channel, num_symbols = 1000, noise_mean = 0, noise_var = 1):
		random_symbols = np.random.randint(transmitter.M, size = num_symbols)
		titleText = ('Eye, TX : ' + transmitter.name + ' , CH : ' + channel.name + 
						'RX: ' + self.name + ' , NoiseVar: %.4f' % noise_var)
		t, modulated_signal = transmitter.transmit_symbols(random_symbols)
		t, transmitted_signal = channel.transmit_signal(modulated_signal)
		transmitted_signal = channel.add_awgn(transmitted_signal, mean = noise_mean, var = noise_var)
		t, rec_signal = self.receive_signal(transmitted_signal)

		N = int(num_symbols * self.__Fs)
		t = t[:N+1]
		rec_signal = rec_signal[:N+1]
		myplt.eye_diagram_plot(t, rec_signal, self.__T_p, eye_N=int(2 * self.__T_p * self.__Fs), titleText = titleText)

class MatchedReceiver(Receiver):
	def __init__(self, transmitter):
		name = transmitter.name + ' Matched'
		t, h_t, T_p, Fs = transmitter.get_matched_filter_response()
		Ts_optimal = T_p # b/c match filter
		super().__init__(h_t, t, T_p, Fs, Ts_optimal, name = name)


if __name__ == "__main__":
	print('This is the class definition - run testReceiver to test it!')
