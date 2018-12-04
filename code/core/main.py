import os
import CommsSystem as comms
import Transmitter as tx
import Channel as chnl
import Receiver as rx
import Equalizer as eqz
import ImageProcessor as ip
import numpy as np
from matplotlib import pyplot as plt

IMG_IN_PATH = os.path.join(os.getcwd(), '../../images_in')
IMG_OUT_PATH = os.path.join(os.getcwd(), '../../images_out')


# works on only 1 file if DEBUG is true - the first file in directory
DEBUG = False
ext = ('.jpg', '.jpeg', '.png',)
image_files = [fname for fname in os.listdir(IMG_IN_PATH) if fname.lower().endswith(ext)]
image_files = image_files[:1] if DEBUG else image_files
image_files = [os.path.join(IMG_IN_PATH, file) for file in image_files]


if __name__ == '__main__':
	# image_processor definition
	block_size = 8
	num_quant_bits = 8
	dct_ip = ip.LinearDCTImageProcessor(block_size, num_quant_bits)
	
	# transmitter definition
	T_pulse = 1	# sec
	Fs = 32		# samples/sec in pulse representation
	alpha = 0.5
	K = 4
	noise_var = 0.0001
	#transmitter = tx.HalfSineTransmitter(T_pulse, Fs)
	transmitter = tx.SRRCTransmitter(alpha, T_pulse, Fs, K)

	# channel definition
	h_ch = np.array([1, 1/2, 3/4, -2/7])
	channel = chnl.Channel(h_ch, np.ones(1), Fs, T_pulse, 'Test')
	#channel = chnl.IndoorChannel(Fs, T_pulse)
	#channel = chnl.OutdoorChannel(Fs, T_pulse)
	
	# receiver definition
	receiver = rx.MatchedReceiver(transmitter)

	# equalizer definition
	#equalizer = eqz.ZFEqualizer(channel, Fs, T_pulse)
	equalizer = eqz.MMSEEqualizer(channel, Fs, T_pulse, noise_var = noise_var)


	comms_sys = comms.CommsSystem(dct_ip, transmitter, 
								  channel, receiver, equalizer)

	for file in image_files:
		img_basename = os.path.splitext(os.path.basename(file))[0]
		comms_sys.run_simulation(file, noise_var = noise_var,
									img_basename = img_basename, 
									VERBOSE = True, DISPLAYED = True, 
									EYES = False, SAVED = False)

	plt.show()