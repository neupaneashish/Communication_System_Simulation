import sys
sys.path.append('../core')
sys.path.append('../utils')

import Transmitter as tx
import Channel as chnl
import Receiver as rx 
import numpy as np
from matplotlib import pyplot as plt
import PlotUtils as myplt

def test_receiver_response(receiver, titleText = None, SAVED = False):
	plt.figure()
	receiver.plot_freq_response(titleText = titleText)
	if SAVED:
		myplt.save_current('{}_receiver_freq_response.png'.format(receiver.name))

	plt.figure()
	receiver.plot_impulse_response(titleText = titleText)
	if SAVED:
		myplt.save_current('{}_receiver_impulse_response.png'.format(receiver.name))

def test_reception_no_channel(receiver, transmitter, bits, SAVED = False):
	t_mod, mod_signal = transmitter.transmit_bits(bits)
	t_rec, rec_signal = receiver.receive_signal(mod_signal)

	samp_signal = receiver.sample_sig_to_bits(t_rec, rec_signal)

	plt.figure()
	plt.subplot(2,2,1)
	myplt.signal_plot(np.arange(np.size(bits)), bits, discrete = True, 
						titleText = 'Transmitter: %s, Receiver: %s' % (transmitter.name, receiver.name),
						xText = 'n', yText = 'b[n]')
	plt.subplot(2,2,2)
	myplt.signal_plot(t_mod, mod_signal, titleText = None)

	plt.subplot(2,2,3)
	myplt.signal_plot(t_rec, rec_signal, titleText = None)

	plt.subplot(2,2,4)
	myplt.signal_plot(np.arange(np.size(samp_signal)), samp_signal, discrete = True, 
						xText = 'n', yText = 'r[n]', titleText = None)

def test_reception_no_noise(receiver, transmitter, channel, bits, titleText = None, SAVED = False):
	t_mod, mod_signal = transmitter.transmit_bits(bits)
	t_tx, tx_signal = channel.transmit_signal(mod_signal)
	t_rec, rec_signal = receiver.receive_signal(tx_signal)

	samp_signal = receiver.sample_sig_to_bits(t_rec, rec_signal)

	if titleText is None:
		titleText = 'TX: %s, CH: %s, RX: %s' % (transmitter.name, channel.name, receiver.name)

	plt.figure()
	plt.subplot(2,2,1)
	myplt.signal_plot(np.arange(np.size(bits)), bits, discrete = True, 
						titleText = titleText,
						xText = 'n', yText = 'b[n]')
	plt.subplot(2,2,2)
	myplt.signal_plot(t_tx, tx_signal, titleText = None)

	plt.subplot(2,2,3)
	myplt.signal_plot(t_rec, rec_signal, titleText = None)

	plt.subplot(2,2,4)
	myplt.signal_plot(np.arange(np.size(samp_signal)), samp_signal, discrete = True, 
						xText = 'n', yText = 'r[n]', titleText = None)


def test_sampling(receiver, transmitter, channel, bits, titleText = None, SAVED = False):
	t_mod, mod_signal = transmitter.transmit_bits(bits)
	t_tx, tx_signal = channel.transmit_signal(mod_signal)
	t_rec, rec_signal = receiver.receive_signal(tx_signal)

	samp_signal = receiver.sample_sig_to_bits(t_rec, rec_signal)
	if titleText is None:
		titleText = 'TX: %s, CH: %s, RX: %s' % (transmitter.name, channel.name, receiver.name)

	plt.figure()
	plt.subplot(2,1,1)
	myplt.signal_plot(np.arange(np.size(bits)), bits, discrete = True, 
						titleText = titleText,
						xText = 'n', yText = 'b[n]')
	plt.subplot(2,1,2)
	myplt.signal_plot(np.arange(np.size(samp_signal)), samp_signal, discrete = True, 
						xText = 'n', yText = 'r[n]', titleText = None)


def test_eye_diagram(receiver, transmitter, channel, noise_var = 0.01, SAVED = False):
	plt.figure()
	receiver.plot_eye_diagram(transmitter, channel, noise_var = noise_var)

if __name__ == "__main__":
	T_pulse = 1	# sec
	Fs = 32		# samples/sec per pulse
	alpha = 0.5
	K = 4
	noise_var = 0.001

	hs_tx = tx.HalfSineTransmitter(T_pulse, Fs)
	hs_rx = rx.MatchedReceiver(hs_tx)

	srrc_tx = tx.SRRCTransmitter(alpha, T_pulse, Fs, K)
	srrc_rx = rx.MatchedReceiver(srrc_tx)

	# channel impulse responses - modeled as an LTI system with finite impulse response 
	h_test = np.array([1, 1/2, 3/4, -2/7])
	ch_test = chnl.Channel(h_test, np.ones(1), Fs, T_pulse, 'Test')


	transmitters = [hs_tx, srrc_tx]
	receivers = [hs_rx, srrc_rx]

	random_bits = np.random.randint(2, size = 10)	# to test transmission

	for receiver, transmitter in zip(receivers, transmitters):
		print('Working on- Receiver: ', receiver.name, 
				', Transmitter: ', transmitter.name, ' ...')
		#test_receiver_response(receiver)
		test_reception_no_channel(receiver, transmitter, random_bits)
		test_reception_no_noise(receiver, transmitter, ch_test, random_bits)
		#test_eye_diagram(receiver, transmitter, ch_test, noise_var = noise_var)
		#test_sampling(receiver, transmitter, ch_test, random_bits)

	plt.show()