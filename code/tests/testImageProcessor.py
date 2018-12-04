import sys
sys.path.append('../core')
sys.path.append('../utils')

import os
import ImageProcessor as ip
import numpy as np
from matplotlib import pyplot as plt
import PlotUtils as myplt


IMG_IN_PATH = os.path.join(os.getcwd(), '../../images_in')
IMG_OUT_PATH = os.path.join(os.getcwd(), '../../images_out')


# works on only 1 file if DEBUG is true - the first file in directory
DEBUG = False
ext = ('.jpg', '.jpeg', '.png',)
image_files = [fname for fname in os.listdir(IMG_IN_PATH) if fname.lower().endswith(ext)]
image_files = image_files[:1] if DEBUG else image_files
image_files = [os.path.join(IMG_IN_PATH, file) for file in image_files]

# displays output image if true
DISPLAYED = False
# saves output image if true
SAVED = False

def test_bin_random(dct_ip, img_size = (16, 16,)):
	img = np.random.randint(2, size = img_size)
	plt.figure()
	myplt.show_image(img, titleText = 'Binary random image')

	bits, maxval, minval = dct_ip.image_to_bits(img)
	img = dct_ip.bits_to_image(bits, 1, np.shape(img), maxval, minval)
	plt.figure()
	myplt.show_image(img, titleText = 'Binary random image decoded')

def test_8bit_random(dct_ip, img_size = (32, 32,)):
	img = np.random.randint(2**8, size = img_size)
	plt.figure()
	myplt.show_image(img, titleText = '8-bit random image')

	bits, maxval, minval = dct_ip.image_to_bits(img)
	
	img = dct_ip.bits_to_image(bits, 8, np.shape(img), maxval, minval)
	plt.figure()
	myplt.show_image(img, titleText = '8-bit random image decoded')

def test_image(dct_ip, filename, img_basename = 'sample'):
	img_in, dim = dct_ip.read_image(filename, FITSIZE = True)
	plt.figure()
	myplt.show_image(img_in, titleText = 'Image input : %s' % img_basename)
	
	bits, maxval, minval = dct_ip.image_to_bits(img_in)
	
	px_n_bits = np.ceil(np.log2(np.max(img_in)))
	print('num bits per px: ', px_n_bits)
	img_out = dct_ip.bits_to_image(bits, px_n_bits, dim, maxval, minval)
	plt.figure()
	myplt.show_image(img_out, titleText = 'Image output : %s' % img_basename)

def test_pika(dct_ip):
	print('#################################################')
	print('Converting image to bits and back: pika.jpg')

	test_image(dct_ip, '../../images_in/pika.jpg', img_basename = 'pika')
		
	print('Done with pika!')
	print('################################################# \n')

if __name__ == '__main__':
	N_bk = 4
	N_bits = 8

	dct_lin_ip = ip.LinearDCTImageProcessor(N_bk, N_bits)

	# test with random matrices
	#test_bin_random(dct_lin_ip)
	#test_8bit_random(dct_lin_ip, img_size = (16, 16,))
	#test_8bit_random(dct_lin_ip, img_size = (N_bk*2,N_bk*2))
	#test_8bit_random(dct_lin_ip, img_size = (N_bk*10, N_bk * 10,))
	#test_8bit_random(dct_lin_ip, img_size = (N_bk*3, N_bk*2,))
	#test_8bit_random(dct_lin_ip, img_size = (N_bk*3, N_bk*4,))
	
	#test_pika(dct_lin_ip)

	# test with actual images
	for file in image_files:
		img_basename = os.path.splitext(os.path.basename(file))[0]
		print('#################################################')
		print('Converting image to bits and back: ', img_basename, ' in ', file)

		test_image(dct_lin_ip, file, img_basename = img_basename)
		
		if SAVED:
			print('Will save to ', os.path.join(IMG_OUT_PATH, '{}_out.png'.format(img_basename)))
		print('Done with ', img_basename, '!')
		print('################################################# \n')

	print('Done with all files in ', IMG_IN_PATH, '!')
	plt.show()
