import numpy as np
import skimage as sk
import matplotlib.pyplot as plt
import os
from src import denoise
from src import compression


if __name__ == "__main__":

    """ IMAGES """
    # img = sk.data.camera() # Load image
    original_img_filename = "img/panda.jpg"
    img = sk.io.imread(original_img_filename)

    """ DENOISING """

    noise_img = denoise.noise_img(img, 30000) # Add noise to image
    denoised_img = denoise.denoise_img(noise_img) # Denoise image using Fourier transform
    denoised_blur = denoise.denoise_blur(noise_img) # Denoise image using Gaussian blur

    _, ax = plt.subplots(2, 2, figsize=(12, 8))

    ax[0][0].imshow(img, cmap="gray")
    ax[0][0].set_title("Original image")
    ax[0][1].imshow(noise_img, cmap="gray")
    ax[0][1].set_title("Noisy image")
    ax[1][0].imshow(denoised_img, cmap="gray")
    ax[1][0].set_title("Denoised image (Fourier)")
    ax[1][1].imshow(denoised_blur, cmap="gray")
    ax[1][1].set_title("Denoised image (Gaussian blur)")

    denoise_fourier_mse = sk.metrics.mean_squared_error(img.flatten(), denoised_img.flatten()) / img.max()
    denoise_blur_mse = sk.metrics.mean_squared_error(img.flatten(), denoised_blur.flatten()) / img.max()

    print(f"Denoise Fourier MSE: {denoise_fourier_mse}")
    print(f"Denoise blur MSE: {denoise_blur_mse} \n")

    """ COMPRESSION """

    comp_img, spectrum = compression.compress_fourier_img(img, 0.9)
    comp_wav_img, spectrum_wav = compression.compress_wavelet_img(img, 0.9)

    fourier_filename = "img/img_compressed_fourier.jpg"
    wavelet_filename = "img/img_compressed_wavelet.jpg"
    #sk.io.imsave(fourier_filename, comp_img)
    #sk.io.imsave(wavelet_filename, comp_wav_img)

    original_stats = os.stat(original_img_filename)
    fourier_stats = os.stat(fourier_filename)
    wavelet_stats = os.stat(wavelet_filename)

    print("Fourier img file size reduction rate: ", fourier_stats.st_size / original_stats.st_size)
    print("Wavelet img file size reduction rate: ", wavelet_stats.st_size / original_stats.st_size, "\n")

    """ PLOTTING """
    fig1, ax1 = plt.subplots(1, 3, figsize=(12, 8))

    ax1[0].imshow(img, cmap="gray")
    ax1[0].set_title("Original image")
    ax1[1].imshow(spectrum, cmap="gray")
    ax1[1].set_title("Compressed data (spectrum)")
    ax1[2].imshow(comp_img, cmap="gray")
    ax1[2].set_title("Decompressed image (Fourier)")
    fig1.suptitle("Fourier compression")

    fig2, ax2 = plt.subplots(1, 3, figsize=(12, 8))

    ax2[0].imshow(img, cmap="gray")
    ax2[0].set_title("Original image")
    ax2[1].imshow(spectrum_wav, cmap="gray")
    ax2[1].set_title("Compressed data (spectrum)")
    ax2[2].imshow(comp_wav_img, cmap="gray")
    ax2[2].set_title("Decompressed image (wavelet)")
    fig2.suptitle("Wavelet compression")

    """ METRICS """

    error = sk.metrics.mean_squared_error(img.flatten(), comp_img.flatten()) / img.max()
    wav_error = sk.metrics.mean_squared_error(img.flatten(), comp_wav_img.flatten()) / img.max()

    print(f"Fourier MSE: {error}")
    print(f"Wavelet MSE: {wav_error}")

    plt.show()
