import sys
sys.path.append('../core')
sys.path.append('../utils')

import Transmitter as tx
import numpy as np
from matplotlib import pyplot as plt
import PlotUtils as myplt

def test_pulse_shape_hs(hs_tx, SAVED = False):
	plt.figure()
	hs_tx.plot_impulse_response()
	if SAVED:
		myplt.save_current('tx_%s_impulse.png' % hs_tx.name, 'PLOT')

	plt.figure()
	hs_tx.plot_freq_response()
	if SAVED:
		myplt.save_current('tx_%s_freq.png' % hs_tx.name, 'PLOT')

def test_pulse_shape_srrc(srrc_tx, K, alpha, SAVED = False):
	plt.figure()
	for transmitter in srrc_tx:
		transmitter.plot_impulse_response(titleText = 'Impulse response SRRC pulse')
	plt.legend(['K=%g, alpha=%.2f' % (k, al) for k in K for al in alpha])
	if SAVED:
		myplt.save_current('tx_SRRC_impulse.png', 'PLOT')

	plt.figure()
	for transmitter in srrc_tx:
		transmitter.plot_freq_response(titleText = 'Frequency response SRRC pulse')
	plt.legend(['K=%g, alpha=%.2f' % (k, al) for k in K for al in alpha])
	if SAVED:
		myplt.save_current('tx_SRRC_freq.png', 'PLOT')

def test_modulation(transmitter, bits, Fs , SAVED = False):
	t, modulated_signal = transmitter.transmit_bits(bits)
	
	plt.figure()
	plt.subplot(2,1,1)
	myplt.signal_plot(np.arange(np.size(bits)), bits, discrete = True, 
						titleText = 'Random signal and its %s modulated signal' % transmitter.name,
						xText = 'n', yText = 'b[n]')
	plt.subplot(2,1,2)
	myplt.signal_plot(t, modulated_signal, titleText = None)
	if SAVED:
		myplt.save_current('tx_%s_mod.png' % transmitter.name, 'PLOT')

	plt.figure()
	myplt.bode_plot(t, modulated_signal, titleText = 'Spectrum of modulated signal %s' % transmitter.name)
	if SAVED:
		myplt.save_current('tx_%s_mod_freq.png' % transmitter.name, 'PLOT')

def test_eye_diagram(transmitter, SAVED = False):
	plt.figure()
	transmitter.plot_eye_diagram()
	
	if SAVED:
		myplt.save_current('eye_{}.png'.format(transmitter.name))

if __name__ == "__main__":

	'''
		Q1) Plot impulse and freq response for both transmitters
			Try changing alpha and K
	'''
	T_pulse = 1	# sec
	Fs = 32		# samples/sec per pulse
	alpha = [0.25, 0.5]
	K = [6, 4] 
	
	hs_tx = tx.HalfSineTransmitter(T_pulse, Fs)
	test_pulse_shape_hs(hs_tx, SAVED = True)
	
	srrc_txs = [tx.SRRCTransmitter(al, T_pulse, Fs, k) for k in K for al in alpha]
	test_pulse_shape_srrc(srrc_txs, K, alpha, SAVED = True)
	

	'''
		Q 2,3,4) Modulated signal and it's spectrum for 10 random bits & eye diagram
	'''
	K = [k for k in K for al in alpha]
	alpha = [al for k in K for al in alpha]
	random_bits = np.random.randint(2, size = 10)
	srrc_tx = srrc_txs[-1]

	transmitters = [hs_tx, srrc_txs[-1], srrc_txs[0]]

#	for transmitter in transmitters:
#		test_modulation(transmitter, random_bits, Fs, SAVED = True)
		#test_eye_diagram(transmitter, SAVED = False)	
	


	plt.show()

		