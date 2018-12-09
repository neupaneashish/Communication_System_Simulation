import sys
sys.path.append('../utils')

import numpy as np
from numpy import fft as ft
from scipy import signal as sp 
import PlotUtils as myplt
from constants import PAMCoeffs as coeffs
from matplotlib import pyplot as plt 

class Equalizer():

	def __init__(self, h_t, t, Fs, T_pulse, name = 'eq'):
		self.__t = t
		self.__h_t = h_t
		self.__Fs = Fs
		self.__T_p = T_pulse
		self.__type = name

	@property
	def name(self):
		return self.__type
	

	def plot_impulse_response(self, titleText = None):		
		if titleText is None:
			titleText = 'Impulse Response ' + self.__type + ' Equalizer'
		myplt.signal_plot(self.__t, self.__h_t, yText = 'h(t)', titleText = titleText, discrete = True)


	def plot_freq_response(self, titleText = None):		
		if titleText is None:
			titleText = 'Frequency Response ' + self.__type + ' Equalizer'
		myplt.bode_plot(self.__t, self.__h_t, titleText = titleText)

	def equalize_signal(self, signal):
		eq_signal = np.convolve(signal, self.__h_t, mode = 'full')
		
		# donot truncate delayed part here just t < 0 - truncation code commented out
		start = np.argwhere(np.abs(self.__t) <= 1/(2 *self.__Fs)).flatten()[0]
		eq_signal = eq_signal[start:]
		#pos_time0 = np.argwhere(np.abs(self.__t) <= 1/(2 *self.__Fs)).flatten()[0]
		pos_time0 = 0
		t = np.arange(-pos_time0/self.__Fs,(np.size(eq_signal)-pos_time0)/self.__Fs, 1/self.__Fs)
		
		# and truncate more if first sample of h_t is later than t = 0
		start = np.argwhere(np.abs(self.__h_t) >= 1e-8).flatten()[0]
		t = t[start:]
		eq_signal = eq_signal[start:]

		return t, eq_signal

	def plot_eye_diagram(self, transmitter, channel, receiver, num_symbols = 1000, noise_mean = 0, noise_var = 1):
		random_symbols = np.random.randint(transmitter.M, size = num_symbols)
		titleText = ('Eye, TX : ' + transmitter.name + ' , CH : ' + channel.name + 
						'RX: ' + receiver.name + 'EQ: ' + self.name +
						 ' , NoiseVar: %.4f' % noise_var)
		t, modulated_signal = transmitter.transmit_symbols(random_symbols)
		t, transmitted_signal = channel.transmit_signal(modulated_signal)
		transmitted_signal = channel.add_awgn(transmitted_signal, mean = noise_mean, var = noise_var)
		t, rec_signal = receiver.receive_signal(transmitted_signal)
		t, eq_signal = self.equalize_signal(rec_signal)

		N = int(num_symbols * self.__Fs)
		t = t[:N+1]
		eq_signal = eq_signal[:N+1]
		myplt.eye_diagram_plot(t, eq_signal, self.__T_p, eye_N=int(2 * self.__T_p * self.__Fs), titleText = titleText)

class ZFEqualizer(Equalizer):
	def __init__(self, channel, Fs, T_pulse):
		# filter coeff are the coeffs of channel inverted - Z(w) = 1 / H(w)
		a, b = channel.get_filter_coefficients()
		
		
		# approximate with a FIR filter
		num_taps = 2**14 -1
		t = np.arange(0, num_taps/Fs, 1/Fs)
		t, h_t = sp.dimpulse(sp.dlti(b, a, dt = 1/Fs), t = t)
		h_t = h_t[0].flatten()[:2**13]
		t = t[:2**13]
		super().__init__(h_t, t, Fs, T_pulse, name = 'ZF')
	
class MMSEEqualizer(Equalizer):
	def __init__(self, channel, Fs, T_pulse, noise_var = 0):
		b_ch, a_ch = channel.get_filter_coefficients()
		
		# approximate with a causal FIR filter
		num_taps = 2**14
				
		w_ch, H_f_ch = sp.freqz(b_ch, a_ch, worN = num_taps, whole = True)

		H_f_eq = np.conj(H_f_ch) / (np.abs(H_f_ch) ** 2 + noise_var)
		
		# plain ifft
		h_t = ft.ifft(H_f_eq)[:2**13]
		t = np.arange(0, np.size(h_t)/Fs, 1/Fs)
		
		'''
		# firls design
		num_taps = num_taps + 1 if num_taps % 2 == 0 else num_taps
		positive_w = np.ix_(w_ch < np.pi)
		h_t = sp.firls(num_taps, w_ch[positive_w], H_f_eq[positive_w])
		'''
		super().__init__(h_t, t, Fs, T_pulse, name = 'MMSE')

if __name__ == "__main__":
	print('This is the class definition - run testReceiver to test it!')
