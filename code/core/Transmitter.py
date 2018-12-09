import sys
sys.path.append('../utils')

import numpy as np
import PlotUtils as myplt
from constants import PAMCoeffs as coeffs


class Transmitter():
	''' f, pulse_f, ''' # - removed - copy back if needed \

	def __init__(self, t, pulse_t,  T_pulse, name = 'tx', PAM_coeff = coeffs.BIN_ANTIPODAL):		
		self.__t = t
		self.__pulse_t = pulse_t
		self.__type = name
		self.__PAM_coeff = PAM_coeff
		self.__T_p = T_pulse
		self.__Fs = 1 / (t[1] - t[0])
		self.__M = np.size(PAM_coeff)

	@property
	def name(self):
		return self.__type
	
	@property
	def M(self):
		return self.__M

	'''
		TRANSMIT THE IMAGE TO THE CHANNEL
	'''
	def transmit_bits(self, bit_stream):
		return self.transmit_symbols(self.bits_to_symbols(bit_stream))	# Binary PAM

	'''
		Convert image into blocks of NxN and convert it to bit stream
	'''
	def bits_to_symbols(self, bit_stream):
		return bit_stream # works only for binary PAM - TODO: include M-ary PAM

	'''
		TRANSMIT A SYMBOL STREAM TO THE CHANNEL
		SYMBOL is a bit in range(0, 2^1) for binary PAM
				and in range(0, M) for M-ary PAM
	'''
	def transmit_symbols(self, symbols):
		symbol_coeffs = np.array([[self.__PAM_coeff[s],  *list(np.zeros(int(self.__Fs) - 1))] for s in symbols]).flatten()
		
		signal = np.convolve(symbol_coeffs, self.__pulse_t, mode='full')

		start = np.argwhere(np.abs(self.__t) <= 1/(2 *self.__Fs)).flatten()[0]
		signal = signal[start:start + np.size(symbol_coeffs) + 1]
		
		t = np.arange(0, np.size(signal)/self.__Fs, 1/self.__Fs)
		
		return t, signal
		
	def get_matched_filter_response(self):
		# assume pulse is symmetric - TODO : interp to get last value for assymmetric pulses
		# h(t) = g(T - t), so t goes from (min - T) to (max - T)
		t = np.arange(-self.__t[-1] + self.__T_p - 1/self.__Fs, 
						-self.__t[0] + self.__T_p,
						1/self.__Fs)
		
		h_t = np.array([self.__pulse_t[0], *list(np.flip(self.__pulse_t[1:], 0))])
		#h_t = np.ones(np.shape(t))
		return t, h_t, self.__T_p, self.__Fs
	

	'''
		Impulse response of the pulse shaping filter
	'''
	def plot_impulse_response(self, label = None):
		print('\tPlotting impulse response ...')
		titleText = 'Impulse Response ' + self.__type + ' pulse shaping filter'
		myplt.signal_plot(self.__t, self.__pulse_t, titleText = titleText, yText = 'h(t)')

	'''
		Frequency response of the pulse shaping filter
	'''
	def plot_freq_response(self, label = None):
		print('\tPlotting frequency response ...')
		titleText = 'Frequency Response ' + self.__type + ' pulse shaping filter'
		myplt.bode_plot(self.__t, self.__pulse_t, titleText = titleText)

	def plot_eye_diagram(self, num_symbols = 1000):
		random_symbols = np.random.randint(self.__M, size = num_symbols)
		titleText = 'Eye Diagram ' + self.__type 
		t, modulated_signal = self.transmit_symbols(random_symbols)
		myplt.eye_diagram_plot(t, modulated_signal, self.__T_p, titleText = titleText)

class SRRCTransmitter(Transmitter):
	def __init__(self, alpha, T_pulse, Fs, K, PAM_coeff = coeffs.BIN_ANTIPODAL):
		N = Fs * T_pulse * 2 * K 

		t = np.arange(-K * T_pulse, K * T_pulse, 1/Fs)
		#t = np.linspace(-K * T_pulse, K * T_pulse, N)
		pulse_t = np.zeros(np.shape(t))
		pulse_t = np.array([ (1 - alpha + 4 * alpha / np.pi) if t[i] == 0 
							else (alpha/np.sqrt(2) * ( (1 + 2/np.pi) * np.sin(np.pi / (4 * alpha)) + 
									 				   (1 - 2/np.pi) * np.cos(np.pi / (4 * alpha)) )
								 ) if np.abs(t[i]) == T_pulse / (4 * alpha) 
								else ( ( np.sin(np.pi * t[i] / T_pulse * (1 - alpha)) + 
								   		 4 * alpha * t[i] / T_pulse * 
									   		np.cos(np.pi * t[i] / T_pulse * ( 1 + alpha)) )
							   		  	/ 
					   		  			( np.pi * t[i] / T_pulse * 
			   		  				    	(1 - (4 * alpha * t[i] / T_pulse) ** 2 ) )
				   		  			)
 							for i in range(np.size(t))
						  ])
		pulse_t = pulse_t / np.sqrt((np.sum(pulse_t ** 2)))
		
		super().__init__(t, pulse_t, T_pulse, name = 'SRRC_K=%g_al=%.2f' % (K, alpha), PAM_coeff = PAM_coeff)

#	def update_pulse(self, alpha, T_pulse, Fs, K, PAM_coeff = super().self.BIN_ANTIPODAL):
#			self.__init__(alpha, T_pulse, Fs, K, PAM_coeff = super().self.BIN_ANTIPODAL)

class HalfSineTransmitter(Transmitter):
	def __init__(self, T_pulse, Fs, PAM_coeff = coeffs.BIN_ANTIPODAL):
		N = Fs * T_pulse 

		t = np.arange(0, T_pulse, 1/Fs)
		pulse_t = np.sin(np.pi / T_pulse * t)
		pulse_t = pulse_t / np.sqrt((np.sum(pulse_t ** 2)))
					
		super().__init__(t, pulse_t, T_pulse, name = 'Half-Sine', PAM_coeff = PAM_coeff)

#	def update_pulse(self, alpha, T_pulse, Fs, K, PAM_coeff = super().self.BIN_ANTIPODAL):
#			self.__init__(alpha, T_pulse, Fs, K, PAM_coeff = PAM_coeff)

if __name__ == "__main__":
	print('This is the class definition - run testTransmitter to test it!')
