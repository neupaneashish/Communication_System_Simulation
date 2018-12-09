

import numpy as np
from scipy import fftpack as ft
import cv2

class ImageProcessor():

	def __init__(self, N, transform, invtransform, num_bits, qct_table = None):
		self.__N = N 	# block size NxN for DCT processing
		self.__transform = transform
		self.__invtransform = invtransform
		self.__num_bits = num_bits
		self.__base = 2	# 	base 2 b/c encoding to bits
		self.__qct_table = np.ones((N, N,)) if qct_table is None else qct_table

	def read_image(self, filename, FITSIZE = False):
		img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
		m, n = np.shape(img)
		if FITSIZE:
			img = img[:m-(m % self.__N), :n-(n % self.__N)]
		return img, np.shape(img)

	def image_to_bits(self, img):
		m, n = np.shape(img)
		all_pixels = []
		maxval = None
		minval = None
		#print('img: ', np.shape(img), img)
		for i in range(0, m, self.__N):
			for j in range(0, n, self.__N):
				block = img[i:i+self.__N, j:j+self.__N]
				block_transformed = self.__transform(block, norm='ortho') / self.__qct_table
				if (maxval is None) or (np.max(block_transformed) > maxval):
					maxval = np.max(block_transformed)
				if (minval is None) or (np.min(block_transformed) < minval):
					minval = np.min(block_transformed)
				all_pixels = all_pixels + list(block_transformed.flatten())
				#print('block: ', np.shape(block), block)
				#print('block transformed: ', np.shape(block_transformed), block_transformed)
	
		#print('all before normalization: ', np.shape(all_pixels), all_pixels)
		all_pixels = (np.array(all_pixels) - minval) / (maxval - minval)
		#print('all pixels: ', np.shape(all_pixels), all_pixels)
		
		bits = self.naive_encode_norm_px_to_bits(all_pixels)
		
		return bits, maxval, minval

	def bits_to_image(self, bits, px_n_bits, dim, maxval, minval):	
		pixels = self.naive_decode_bits_to_px(bits)
		#print('pixels decoded: ', np.shape(pixels), pixels)
		# normalize and scale pixels 
		pixels = pixels / (self.__base**self.__num_bits)
		pixels = (pixels * (maxval - minval)) + minval	

		pixels = np.split(pixels, np.size(pixels) / (self.__N**2))	
		#print('split pixels: ', np.shape(pixels), pixels)
		
		pixels = [self.__invtransform(np.reshape(pixel, (self.__N, self.__N)), norm='ortho') for pixel in pixels]
		#print('pixels trans: ', pixels)

		m, n = dim 
		img = np.zeros(dim)
		for i in range(0, m, self.__N):
			for j in range(0, n, self.__N):
				idx = int((j + i * n/self.__N)/(self.__N))
				#print('i: ', i, 'j:', j, 'numcols: ', n/self.__N, 'idx: ', idx)
				img[i:i+self.__N, j:j+self.__N] = pixels[idx]
	
		img = (img - np.min(img)) / (np.max(img) - np.min(img))
		img = np.round(img * (2**px_n_bits - 1))
		return img.astype(int)

	def naive_decode_bits_to_px(self, bits):
		#print('bits before: ', np.shape(bits))
		pixels = [bits[i:i+self.__num_bits] for i in range(0, np.size(bits), self.__num_bits)]
		#print('bits split: ', np.shape(pixels), pixels)

		pow_base = self.__base**np.arange(self.__num_bits - 1, -1, -1)
		pixels = np.array([np.sum(pixel*pow_base) for pixel in pixels])
		return pixels
	
	def naive_encode_norm_px_to_bits(self, pixels):	# array of pixels in range 0 - 1
		pow_base = self.__base**np.arange(self.__num_bits - 1, -1, -1)
		#print('base powers', pow_base)
		pixels = [int(round(px * (self.__base**self.__num_bits - 1))) for px in pixels]
		#print('pixels encoding: ', np.size(pixels), pixels)
		
		bits = [[(px // pow) % self.__base for pow in pow_base]   for px in pixels]
		bits = np.array(bits).flatten()
		return bits

class NoCompressDCTImageProcessor(ImageProcessor):
	def __init__(self, N_block, N_bits_per_px):
		transform = ft.dctn
		invtransform = ft.idctn
		super().__init__(N_block, transform, invtransform, N_bits_per_px)

class LinearDCTImageProcessor(ImageProcessor):
	def __init__(self, N_block, N_bits_per_px):
		transform = ft.dctn
		invtransform = ft.idctn
		lin_qct_table = np.ones((N_block, N_block,))
		for i in range(N_block//2, N_block):
			for j in range(N_block//2, N_block):
				lin_qct_table[i, j] = 2 * i * j
		super().__init__(N_block, transform, invtransform, N_bits_per_px, qct_table = lin_qct_table)


if __name__ == "__main__":
	print('This is the class definition - run testImageProcessor to test it!')
