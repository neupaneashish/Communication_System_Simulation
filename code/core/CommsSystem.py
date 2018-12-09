import sys
sys.path.append('../core')
sys.path.append('../utils')
import os
import Transmitter as tx
import Channel as chnl
import Receiver as rx
import Equalizer as eqz
import ImageProcessor as ip
import numpy as np
from matplotlib import pyplot as plt
import PlotUtils as myplt

import cv2
IMG_OUT_PATH = os.path.join(os.getcwd(), '../../images_out')


class CommsSystem():
	def __init__(self, img_processor, transmitter, 
				channel, receiver, equalizer):
		self.__ip = img_processor
		self.__tx = transmitter
		self.__ch = channel
		self.__rx = receiver
		self.__eq = equalizer
		
	def run_simulation(self, filename, noise_var = 0, img_basename = 'sample', 
						DISPLAYED = False, 
						EYES = False, SAVED = False):
		pass

	def run_simulation_gray(self, filename, noise_var = 0, img_basename = 'sample', 
						DISPLAYED = False, EYES = False, SAVED = False):
		print('\n#################################################')
		print('Reading file as grayscale image: ', img_basename)
		img_in, dim = self.__ip.read_image_gray(filename, FITSIZE = True)
		if DISPLAYED:
			plt.figure()
			myplt.show_image(img_in, titleText = '%s input' % img_basename)
		
		img_out = self.transmit_image(img_in, dim, noise_var = noise_var, EYES = EYES, SAVED = SAVED)
		if DISPLAYED:
			plt.figure()
			myplt.show_image(img_out, titleText = '%s, tx: %s, ch: %s, eq: %s, noise:%.4f' % 
					(img_basename, self.__tx.name, self.__ch.name, self.__eq.name, noise_var))
		if SAVED:
			filename = ('%s_tx_%s_ch_%s_eq_%s_ns_%.4f.png' % 
								(img_basename, self.__tx.name, self.__ch.name, self.__eq.name, noise_var))
			ret = cv2.imwrite(os.path.join(IMG_OUT_PATH, filename), img_out)
			if ret:
				print('Output image saved.')
			else:
				print('Couldn"t save image output :(')

		print('Done with ', img_basename, '!')              
		print('#################################################\n')

	def transmit_image(self, img, dim, noise_var = 0, EYES = False, SAVED = False):
		print('Preprocessing image ...')
		bits, maxval, minval = self.__ip.image_to_bits(img)
	
		print('Modulating bits ...')
		t_mod, mod_signal = self.__tx.transmit_bits(bits)
		N_sig = len(mod_signal)


		print('Transmitting signal ...')
		t_tx, tx_signal = self.__ch.transmit_signal(mod_signal)
		tx_signal = self.__ch.add_awgn(tx_signal, var = noise_var)

		print('Receiving signal ...')       
		t_rec, rec_signal = self.__rx.receive_signal(tx_signal)

		print('Equalizing signal ...')              
		t_eq, eq_signal = self.__eq.equalize_signal(rec_signal)

		print('Sampling signal ...')                
		samp_signal = self.__rx.sample_sig_to_bits(t_eq, eq_signal)[:np.size(bits)]

		print('Postprocessing image ...')               
		px_n_bits = np.ceil(np.log2(np.max(img)))
		img_out = self.__ip.bits_to_image(samp_signal, px_n_bits, dim, maxval, minval)

		if EYES: 
			print('Plotting eye diagrams ...')               
			# eye after transmitter
			plt.figure()
			myplt.eye_diagram_plot(t_mod, mod_signal, 1, eye_N = 32, titleText = 'Eye Diagram after TX: %s' % self.__tx.name )
			if SAVED:
				myplt.save_current('eye_tx_%s.png' % self.__tx.name, 'PLOT')

			# eye after channel and noise
			plt.figure()
			myplt.eye_diagram_plot(t_tx[:N_sig], tx_signal[:N_sig], 1, eye_N = 32, titleText = 'Eye Diagram after CH: %s Noise: %.4f' % (self.__ch.name, noise_var) )
			if SAVED:
				myplt.save_current('eye_tx_%s_ch_%s_ns_%.4f.png' % (self.__tx.name, self.__ch.name, noise_var), 'PLOT')

			# eye after matched filter
			plt.figure()
			myplt.eye_diagram_plot(t_rec[:N_sig], rec_signal[:N_sig], 1, eye_N = 2*32, titleText = 'Eye Diagram after RX: %s Noise: %.4f' % (self.__rx.name, noise_var) )
			if SAVED:
				myplt.save_current('eye_rx_%s_ch_%s_ns_%.4f.png' % (self.__rx.name, self.__ch.name, noise_var), 'PLOT')

			# eye after equalizer
			plt.figure()
			myplt.eye_diagram_plot(t_eq[:N_sig], eq_signal[:N_sig], 1, eye_N = 2*32, titleText = 'Eye Diagram after EQ: %s Noise: %.4f' % (self.__eq.name, noise_var) )
			if SAVED:
				myplt.save_current('eye_rx_%s_ch_%s_eq_%s_ns_%.4f.png' % (self.__rx.name, self.__ch.name, self.__eq.name, noise_var), 'PLOT')
		return img_out
	


if __name__ == "__main__":
	print('This is the class definition - use main.py to run sample simulation!')
