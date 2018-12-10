import os
import CommsSystem as comms
import Transmitter as tx
import Channel as chnl
import Receiver as rx
import Equalizer as eqz
import ImageProcessor as ip
import numpy as np
from matplotlib import pyplot as plt
import cv2

IMG_IN_PATH = os.path.join(os.getcwd(), '../../images_in')
IMG_OUT_PATH = os.path.join(os.getcwd(), '../../images_out')


# works on only n files - set to None to do all
N_FILES = 1
SAVE_OUTPUT = True
ext = ('.jpg', '.jpeg', '.png',)
image_files = [fname for fname in os.listdir(IMG_IN_PATH) if fname.lower().endswith(ext)]
image_files = image_files[:N_FILES] if N_FILES is not None else image_files
image_files = [os.path.join(IMG_IN_PATH, file) for file in image_files]

image_files = [os.path.join(IMG_IN_PATH, 'cameraman_gray.png')]

def main():
    block_size = 8
    qbits = 8
    T_pulse = 1 # sec
    Fs = 32     # samples/sec in pulse representation
    alpha = 0.5
    K = 4
    noises = [0.002, 0.003, 0.015, 0.02]
    #transmit_camera_with_eye_plots_zf(block_size, num_quant_bits, T_pulse, Fs, K, alpha)
    #transmit_camera_with_eye_plots_mmse(block_size, num_quant_bits, T_pulse, Fs, K, alpha)
    #comms_sys = initialize_hs_zf_testch(block_size, qbits, T_pulse, Fs)
    #comms_sys = initialize_srrc_zf_testch(block_size, qbits, T_pulse, Fs, K, alpha)
    for file in image_files:
        img_basename = os.path.splitext(os.path.basename(file))[0]
        for noise_var in noises:
            comms_systems = [initialize_srrc_zf_testch(block_size, qbits, T_pulse, Fs, K, alpha),
                            initialize_srrc_mmse_testch(block_size, qbits, T_pulse, Fs, K, alpha, noise = noise_var),
                            initialize_hs_zf_testch(block_size, qbits, T_pulse, Fs), 
                            initialize_hs_mmse_testch(block_size, qbits, T_pulse, Fs, noise = noise_var)]
            # comms_sys = initialize_hs_mmse_testch(block_size, qbits, T_pulse, Fs, noise = noise_var)
            # comms_sys = initialize_srrc_mmse_indoorch(block_size, qbits, T_pulse, Fs, K, alpha, noise = noise_var)
            # comms_sys = initialize_srrc_mmse_outdoorch(block_size, qbits, T_pulse, Fs, K, alpha, noise = noise_var)
            for comms_sys in comms_systems:
                comms_sys.run_simulation_gray(file, noise_var = noise_var,
                                        img_basename = img_basename, 
                                        DISPLAYED = False, SAVED = True)
    plt.show()


def transmit_camera_with_eye_plots_mmse(block_size, qbits, T_pulse, Fs, K, alpha, comms_systems = None, noise_var = None):
    if noise_var is not None:
        noises = [noise_var]
    else:
        noises = [0, 0.01]

    filename = os.path.join(IMG_IN_PATH, 'camera-icon.png')
    for noise in noises:
        comms_systems = [initialize_hs_mmse_testch(block_size, qbits, T_pulse, Fs, noise = noise),
                        initialize_srrc_mmse_testch(block_size, qbits, T_pulse, Fs, K, alpha, noise  = noise), 
                        initialize_srrc_mmse_indoorch(block_size, qbits, T_pulse, Fs, K, alpha, noise = noise), 
                        initialize_srrc_mmse_outdoorch(block_size, qbits, T_pulse, Fs, K, alpha, noise = noise),
                        initialize_hs_mmse_testch(block_size, qbits, T_pulse, Fs, noise = 0)]
        for comms_sys in comms_systems:
            comms_sys.run_simulation_gray(filename, noise_var = noise,
                                img_basename = 'camera', DISPLAYED = False, 
                                EYES = True, SAVED = True)  


def transmit_camera_with_eye_plots_zf(block_size, qbits, T_pulse, Fs, K, alpha, comms_systems = None, noise_var = None):
    if comms_systems is not None:
        comms_systems = [comms_systems]
    else:
        comms_systems = [initialize_hs_zf_testch(block_size, qbits, T_pulse, Fs),
                        initialize_srrc_zf_testch(block_size, qbits, T_pulse, Fs, K, alpha)]

    if noise_var is not None:
        noises = [noise_var]
    else:
        noises = [0, 0.01]

    filename = os.path.join(IMG_IN_PATH, 'camera-icon.png')
    for comms_sys in comms_systems:
        for noise in noises:
            comms_sys.run_simulation_gray(filename, noise_var = noise,
                                img_basename = 'camera', DISPLAYED = False, 
                                EYES = True, SAVED = True)  

def initialize_hs_zf_testch(block_size, qbits, T_pulse, Fs):
    dct_ip = ip.NoCompressDCTImageProcessor(block_size, qbits)
    transmitter = tx.HalfSineTransmitter(T_pulse, Fs)
    receiver = rx.MatchedReceiver(transmitter)

    # channel definition
    h_ch = np.array([1, 1/2, 3/4, -2/7])
    channel = chnl.Channel(h_ch, np.ones(1), Fs, T_pulse, 'Test')
    equalizer = eqz.ZFEqualizer(channel, Fs, T_pulse)

    return comms.CommsSystem(dct_ip, transmitter, 
                            channel, receiver, equalizer)

def initialize_hs_mmse_testch(block_size, qbits, T_pulse, Fs, noise = 0):
    dct_ip = ip.NoCompressDCTImageProcessor(block_size, qbits)
    transmitter = tx.HalfSineTransmitter(T_pulse, Fs)
    receiver = rx.MatchedReceiver(transmitter)

    # channel definition
    h_ch = np.array([1, 1/2, 3/4, -2/7])
    channel = chnl.Channel(h_ch, np.ones(1), Fs, T_pulse, 'Test')
    equalizer = eqz.MMSEEqualizer(channel, Fs, T_pulse, noise_var = noise)

    return comms.CommsSystem(dct_ip, transmitter, 
                            channel, receiver, equalizer)

def initialize_hs_mmse_indoorch(block_size, qbits, T_pulse, Fs, noise = 0):
    dct_ip = ip.NoCompressDCTImageProcessor(block_size, qbits)
    transmitter = tx.HalfSineTransmitter(T_pulse, Fs)
    receiver = rx.MatchedReceiver(transmitter)
    channel = chnl.IndoorChannel(Fs, T_pulse)
    equalizer = eqz.MMSEEqualizer(channel, Fs, T_pulse, noise_var = noise)

    return comms.CommsSystem(dct_ip, transmitter, 
                            channel, receiver, equalizer)

def initialize_hs_mmse_outdoorch(block_size, qbits, T_pulse, Fs, noise = 0):
    dct_ip = ip.NoCompressDCTImageProcessor(block_size, qbits)
    transmitter = tx.HalfSineTransmitter(T_pulse, Fs)
    receiver = rx.MatchedReceiver(transmitter)
    channel = chnl.OutdoorChannel(Fs, T_pulse)
    equalizer = eqz.MMSEEqualizer(channel, Fs, T_pulse, noise_var = noise)

    return comms.CommsSystem(dct_ip, transmitter, 
                            channel, receiver, equalizer)

def initialize_srrc_zf_testch(block_size, qbits, T_pulse, Fs, K, alpha):
    dct_ip = ip.NoCompressDCTImageProcessor(block_size, qbits)
    transmitter = tx.SRRCTransmitter(alpha, T_pulse, Fs, K)
    receiver = rx.MatchedReceiver(transmitter)
    h_ch = np.array([1, 1/2, 3/4, -2/7])
    channel = chnl.Channel(h_ch, np.ones(1), Fs, T_pulse, 'Test')
    equalizer = eqz.ZFEqualizer(channel, Fs, T_pulse)

    return comms.CommsSystem(dct_ip, transmitter, 
                            channel, receiver, equalizer)


def initialize_srrc_mmse_testch(block_size, qbits, T_pulse, Fs, K, alpha, noise = 0):
    dct_ip = ip.NoCompressDCTImageProcessor(block_size, qbits)
    transmitter = tx.SRRCTransmitter(alpha, T_pulse, Fs, K)
    receiver = rx.MatchedReceiver(transmitter)
    h_ch = np.array([1, 1/2, 3/4, -2/7])
    channel = chnl.Channel(h_ch, np.ones(1), Fs, T_pulse, 'Test')
    equalizer = eqz.MMSEEqualizer(channel, Fs, T_pulse, noise_var = noise)

    return comms.CommsSystem(dct_ip, transmitter, 
                            channel, receiver, equalizer)

def initialize_srrc_mmse_indoorch(block_size, qbits, T_pulse, Fs, K, alpha, noise = 0):
    dct_ip = ip.NoCompressDCTImageProcessor(block_size, qbits)
    transmitter = tx.SRRCTransmitter(alpha, T_pulse, Fs, K)
    receiver = rx.MatchedReceiver(transmitter)
    channel = chnl.IndoorChannel(Fs, T_pulse)
    equalizer = eqz.MMSEEqualizer(channel, Fs, T_pulse, noise_var = noise)

    return comms.CommsSystem(dct_ip, transmitter, 
                            channel, receiver, equalizer)

def initialize_srrc_mmse_outdoorch(block_size, qbits, T_pulse, Fs, K, alpha, noise = 0):
    dct_ip = ip.NoCompressDCTImageProcessor(block_size, qbits)
    transmitter = tx.SRRCTransmitter(alpha, T_pulse, Fs, K)
    receiver = rx.MatchedReceiver(transmitter)
    channel = chnl.OutdoorChannel(Fs, T_pulse)
    equalizer = eqz.MMSEEqualizer(channel, Fs, T_pulse, noise_var = noise)

    return comms.CommsSystem(dct_ip, transmitter, 
                            channel, receiver, equalizer)


if __name__ == '__main__':
    main()
