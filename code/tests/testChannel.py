import sys
sys.path.append('../core')
sys.path.append('../utils')

import Transmitter as tx
import Channel as chnl
import numpy as np
from matplotlib import pyplot as plt
import PlotUtils as myplt

def test_channel_response(channel, titleText = None, SAVED = False):
	plt.figure()
	channel.plot_freq_response(titleText = titleText)
	if SAVED:
		myplt.save_current('ch_{}_freq.png'.format(channel.name), 'PLOT')

	plt.figure()
	channel.plot_impulse_response(titleText = titleText)
	if SAVED:
		myplt.save_current('ch_{}_impulse.png'.format(channel.name), 'PLOT')

def test_transmission(channel, transmitter, bits, SAVED = False):
	t, modulated_signal = transmitter.transmit_bits(bits)
	t, transmitted_signal = channel.transmit_signal(modulated_signal)
	plt.figure()
	plt.subplot(2,1,1)
	myplt.signal_plot(np.arange(np.size(bits)), bits, discrete = True, 
						titleText = 'Transmitter: %s, Channel: %s' % (transmitter.name, channel.name),
						xText = 'n', yText = 'b[n]')
	plt.subplot(2,1,2)
	myplt.signal_plot(t, transmitted_signal, titleText = None)
	if SAVED:
		myplt.save_current('{}_modulated.png'.format(transmitter.name))

	plt.figure()
	myplt.bode_plot(t, transmitted_signal, titleText = 'Spectrum Transmitter: %s, Channel: %s' % (transmitter.name, channel.name))
	if SAVED:
		myplt.save_current('{}_modulated_freq.png'.format(transmitter.name))

def test_eye_diagram(channel, transmitter, SAVED = False):
	plt.figure()
	channel.plot_eye_diagram(transmitter)
	
	if SAVED:
		filename = 'eye_tx_%s_ch_%s.png' % (transmitter.name, channel.name)
		myplt.save_current(filename, 'PLOT')

def test_noise_eye(channel, transmitter, mean = 0, var = 1, SAVED = False):
	plt.figure()
	channel.plot_eye_diagram_after_noise(transmitter, mean = mean, var = var)

	if SAVED:
		filename = 'eye_tx_%s_ch_%s_ns_%.4f.png' % (transmitter.name, channel.name, var)
		myplt.save_current(filename)

if __name__ == "__main__":
	T_pulse = 1	# sec
	Fs = 32		# samples/sec in pulse representation
	alpha = 0.5
	K = 4

	noise_mean = 0
	noise_vars = [0, 0.0001, 0.010]

	hs_tx = tx.HalfSineTransmitter(T_pulse, Fs)
	srrc_tx = tx.SRRCTransmitter(alpha, T_pulse, Fs, K)
	transmitters = [hs_tx, srrc_tx]

	# channel impulse responses - modeled as an LTI system with finite impulse response 
	h_test = np.array([1, 1/2, 3/4, -2/7])
	

	#ch_test = chnl.Channel(h_test, np.ones(1), Fs, T_pulse, 'Test')
	ch_indoor = chnl.IndoorChannel(Fs, T_pulse)
	ch_outdoor = chnl.OutdoorChannel(Fs, T_pulse)
	channels = [ch_indoor, ch_outdoor]
	#channels = [ch_test]

	random_bits = np.random.randint(2, size = 10)	# to test transmission
			
	for channel in channels:
		#test_channel_response(channel, SAVED = True)

		for transmitter in transmitters:
			#test_transmission(channel, transmitter, random_bits)

			test_eye_diagram(channel, transmitter, SAVED = False)
			for noise_var in noise_vars:
				pass
				test_noise_eye(channel, transmitter, mean = noise_mean, var = noise_var, SAVED = False)

	plt.show()