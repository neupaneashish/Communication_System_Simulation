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
		myplt.save_current('{}_channel_freq_response.png'.format(channel.name))

	plt.figure()
	channel.plot_impulse_response(titleText = titleText)
	if SAVED:
		myplt.save_current('{}_channel_impulse_response.png'.format(channel.name))


def test_eye_diagram(channel, transmitter, SAVED = False):
	plt.figure()
	channel.plot_eye_diagram(transmitter)
	
	if SAVED:
		filename = 'eye_%s_%s.png' % (channel.name, transmitter.name)
		myplt.save_current(filename)

def test_noise_eye(channel, transmitter, mean = 0, var = 1, SAVED = False):
	plt.figure()
	channel.plot_eye_diagram_after_noise(transmitter, mean = mean, var = var)

	if SAVED:
		filename = 'eye_%s_%s_noise%.4f.png' % (channel.name, transmitter.name, var)
		myplt.save_current(filename)

if __name__ == "__main__":
	T_pulse = 1	# sec
	Fs = 32		# samples/sec in pulse representation
	alpha = 0.5
	K = 2

	noise_mean = 0
	noise_vars = [0.001, 0.01, 0.1]

	hs_tx = tx.HalfSineTransmitter(T_pulse, Fs)
	srrc_tx = tx.SRRCTransmitter(alpha, T_pulse, Fs, K)
	transmitters = [hs_tx, srrc_tx]

	# channel impulse responses - modeled as an LTI system with finite impulse response 
	h_test = np.array([1, 1/2, 3/4, -2/7])
	b = h_test
	a = np.ones(1)
	

	ch_test = chnl.Channel(h_test, np.ones(1), Fs, T_pulse, 'Test')
	ch_indoor = chnl.IndoorChannel(Fs, T_pulse)
	ch_outdoor = chnl.OutdoorChannel(Fs, T_pulse)
	channels = [ch_test, ch_indoor, ch_outdoor]

	for channel in channels:
		test_channel_response(channel, SAVED = True)

		for transmitter in transmitters:
			test_eye_diagram(channel, transmitter, SAVED = True)

			for noise_var in noise_vars:
				test_noise_eye(channel, transmitter, mean = noise_mean, var = noise_var, SAVED = True)

	plt.show()