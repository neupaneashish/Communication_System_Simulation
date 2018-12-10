import sys
sys.path.append('../core')
sys.path.append('../utils')

import Transmitter as tx
import Channel as chnl
import Receiver as rx
import Equalizer as eqz
import numpy as np
from matplotlib import pyplot as plt
import PlotUtils as myplt

def test_equalizer_response(equalizer, titleText = None, noise = 0, SAVED = False):
	plt.figure()
	equalizer.plot_freq_response(titleText = titleText)
	if SAVED:
		myplt.save_current('%s_eq_freq_ns_%.4f.png' % (equalizer.name, noise), 'PLOT')

	plt.figure()
	equalizer.plot_impulse_response(titleText = titleText)
	if SAVED:
		myplt.save_current('%s_eq_impulse_ns_%.4f.png' % (equalizer.name, noise), 'PLOT')

def test_equalization_only_channel(equalizer, channel, bits, Fs, titleText = None, SAVED = False):
	bits = np.array([[s,  *list(np.zeros(Fs - 1))] for s in bits]).flatten()
	t_bits = np.arange(0, np.size(bits)/Fs, 1/Fs)

	t_tx, signal_tx = channel.transmit_signal(bits)
	t_eq, signal_eq = equalizer.equalize_signal(signal_tx)

	plt.figure()
	plt.subplot(3,1,1)
	myplt.signal_plot(t_bits, bits, discrete = True, 
						titleText = 'Channel: %s, Equalizer: %s' % (channel.name, equalizer.name),
						xText = 'n', yText = 'b[n]')
	plt.subplot(3,1,2)
	myplt.signal_plot(t_tx, signal_tx, titleText = None, discrete = True)

	plt.subplot(3,1,3)
	myplt.signal_plot(t_eq, signal_eq, titleText = None, discrete = True)

def test_equalization_no_noise(equalizer, channel, transmitter, receiver, bits, titleText = None, SAVED = False):
	t_mod, mod_signal = transmitter.transmit_bits(bits)
	t_tx, tx_signal = channel.transmit_signal(mod_signal)
	t_rx, rec_signal = receiver.receive_signal(tx_signal)
	t_eq, eq_signal = equalizer.equalize_signal(rec_signal)

	if titleText is None:
		titleText = 'TX: %s, CH: %s, RX: %s, EQ: %s' % (transmitter.name, channel.name, receiver.name, equalizer.name)
	plt.figure()
	plt.subplot(2,2,1)
	myplt.signal_plot(t_mod, mod_signal, titleText = titleText)

	plt.subplot(2,2,2)
	myplt.signal_plot(t_tx, tx_signal, titleText = None)

	plt.subplot(2,2,3)
	myplt.signal_plot(t_rx, rec_signal, titleText = None)

	plt.subplot(2,2,4)
	myplt.signal_plot(t_eq, eq_signal, titleText = None)

def test_eye_diagram(equalizer, channel, transmitter, receiver, noise_var = 0.01, SAVED = False):
	plt.figure()
	equalizer.plot_eye_diagram(transmitter, channel, receiver, noise_var = noise_var)
	if SAVED:
		myplt.save_current('eye_rx_%s_ch_%s_eq_%s_ns_%.4f.png' % 
							(receiver.name, channel.name, equalizer.name, noise_var),
							'PLOT')


def test_sampling(equalizer, channel, transmitter, receiver, bits, titleText = None, SAVED = False):
	t_mod, mod_signal = transmitter.transmit_bits(bits)
	t_tx, tx_signal = channel.transmit_signal(mod_signal)
	t_rec, rec_signal = receiver.receive_signal(tx_signal)
	t_eq, eq_signal = equalizer.equalize_signal(rec_signal)

	samp_signal = receiver.sample_sig_to_bits(t_eq, eq_signal)[:np.size(bits)]
	if titleText is None:
		titleText = 'TX: %s, CH: %s, RX: %s, EQ: %s' % (transmitter.name, channel.name, receiver.name, equalizer.name)

	plt.figure()
	plt.subplot(2,1,1)
	myplt.signal_plot(np.arange(np.size(bits)), bits, discrete = True, 
						titleText = titleText,
						xText = 'n', yText = 'b[n]')
	plt.subplot(2,1,2)
	myplt.signal_plot(np.arange(np.size(samp_signal)), samp_signal, discrete = True, 
						xText = 'n', yText = 'r[n]', titleText = None)

if __name__ == '__main__':
	T_pulse = 1	# sec
	Fs = 32		# samples/sec in pulse representation
	alpha = 0.5
	K = 4
	noise_var = 0.001

	random_bits = np.random.randint(2, size = 10)	# to test transmission
	# channel impulse responses - modeled as an LTI system with finite impulse response 
	#h_test = np.array([1, 1/2, 3/4, -2/7])
	#ch_test = chnl.Channel(h_test, np.ones(1), Fs, T_pulse, 'Test')

	#ch_test = chnl.IndoorChannel(Fs, T_pulse)
	ch_test = chnl.OutdoorChannel(Fs, T_pulse)

	eq_zf = eqz.ZFEqualizer(ch_test, Fs, T_pulse)
	eq_mmse = eqz.MMSEEqualizer(ch_test, Fs, T_pulse, noise_var = noise_var)
	equalizers = [eq_zf, eq_mmse]

	# transmitter and receiver - 
	#tx_test = tx.HalfSineTransmitter(T_pulse, Fs)
	tx_test = tx.SRRCTransmitter(alpha, T_pulse, Fs, K)
	rx_test = rx.MatchedReceiver(tx_test)

	for equalizer in equalizers:
		print('Working on- Equalizer: ', equalizer.name, 
				', Channel: ', ch_test.name, ' ...')
		
		#test_equalizer_response(equalizer, noise = noise_var, SAVED = True)
		#test_equalization_only_channel(equalizer, ch_test, , random_bits, Fs)
		#test_equalization_no_noise(equalizer, ch_test, tx_test, rx_test, random_bits)
		test_eye_diagram(equalizer, ch_test, tx_test, rx_test, noise_var = noise_var, SAVED=True)
		#test_sampling(equalizer, ch_test, tx_test, rx_test, random_bits)

	plt.show()