import sys
sys.path.append('../core')
sys.path.append('../utils')

import Transmitter as tx
import Channel as chnl
import Receiver as rx
import Equalizer as eqz
import ImageProcessor as ip
import numpy as np
from matplotlib import pyplot as plt
import PlotUtils as myplt

class CommsSystem():
	def __init__(self, img_processor, transmitter, channel, receiver, equalizer):
		self.__ip = img_processor
		self.__tx = transmitter
		self.__ch = channel
		self.__rx = receiver
		self.__eq = equalizer

	def run_simulation(self, filename, noise_var = 0, img_basename = 'sample', 
						VERBOSE = True, DISPLAYED = False, 
						EYES = False, SAVED = False):

		if VERBOSE:
			print('\n#################################################')
			print('Reading file: ', img_basename)
		img_in, dim = self.__ip.read_image(filename, FITSIZE = True)
		if DISPLAYED:
			plt.figure()
			myplt.show_image(img_in, titleText = 'Image input : %s' % img_basename)
	
		if VERBOSE:
			print('Preprocessing image ...')
		bits, maxval, minval = self.__ip.image_to_bits(img_in)
	
		if VERBOSE:
			print('Modulating bits ...')
		t_mod, mod_signal = self.__tx.transmit_bits(bits)

		if VERBOSE:
			print('Transmitting signal ...')
		t_tx, tx_signal = self.__ch.transmit_signal(mod_signal)
		tx_signal = self.__ch.add_awgn(tx_signal, var = noise_var)

		if VERBOSE:
			print('Receiving signal ...')		
		t_rec, rec_signal = self.__rx.receive_signal(tx_signal)

		if VERBOSE:
			print('Equalizing signal ...')				
		t_eq, eq_signal = self.__eq.equalize_signal(rec_signal)

		if VERBOSE:
			print('Sampling signal ...')				
		samp_signal = self.__rx.sample_sig_to_bits(t_eq, eq_signal)[:np.size(bits)]
	
		if VERBOSE:
			print('Postprocessing image ...')				
		px_n_bits = np.ceil(np.log2(np.max(img_in)))
		img_out = self.__ip.bits_to_image(samp_signal, px_n_bits, dim, maxval, minval)
		
		if DISPLAYED:
			plt.figure()
			myplt.show_image(img_out, titleText = 'Image output : %s' % img_basename)

		if SAVED:
			pass

		if EYES: 
			pass

		if VERBOSE:
			print('Done with ', img_basename, '!')				
			print('#################################################\n')

if __name__ == "__main__":
	print('This is the class definition - use main.py to run sample simulation!')
