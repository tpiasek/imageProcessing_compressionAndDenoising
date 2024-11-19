import numpy as np
import matplotlib.pyplot as plt
import pywt
from src import convolution as conv


def noise_img(img, noise_multiplier=200):
    """
    Function that adds noise to the image.

    :param img: Original image
    :return: Noisy image
    """

    img_copy = np.zeros_like(img)

    if len(img.shape) > 2 and img.shape[2] > 1:
        for ch in range(img.shape[2]):
            img_copy[:, :, ch] = noise_img(img[:, :, ch], noise_multiplier)
        return img_copy


    img_fft = np.fft.fft2(img)
    noise = np.random.normal(0, noise_multiplier, img_fft.shape) * np.exp(1j * 2 * np.pi * np.random.random(img_fft.shape))
    # noise = np.random.normal(0, 0.1, img.shape)
    noisy_img = img_fft + noise
    noisy_img = np.fft.ifft2(noisy_img)
    img_copy = np.clip(np.abs(noisy_img), 0, 255).astype(np.uint8)

    return img_copy


def denoise_img(img, threshold_multiplier=0.0005):
    """
    Function that denoises an image using the Fourier transform.
    :param img: Image for denoising [NDArray either grayscale or RGB]
    :param threshold_multiplier: Value for the threshold (between 0 and 1)
    :return: Denoised image [NDArray]
    """

    # If the image if float type, we convert it to 8-bit uint8
    if img.max() <= 1:
        img *= 255
        img = img.astype(np.uint8)
        img = np.clip(img, 0, 255).astype(np.uint8)

    denoised_img_full = np.zeros_like(img)

    _, ax8 = plt.subplots(2, 2, figsize=(12, 8))

    # If the image is RGB, we denoise each channel separately by recursion
    if len(img.shape) > 2 and img.shape[2] > 1:
        for ch in range(img.shape[2]):
            denoised_img_full[:, :, ch] = denoise_img(img[:, :, ch], threshold_multiplier)
        return denoised_img_full

    img_fft = np.fft.fft2(img)  # Apply Fourier transform to the image

    magnitude = np.abs(img_fft)  # Compute the magnitude
    phase = np.angle(img_fft)  # Compute the phase

    freqs = np.fft.fftfreq(len(img))  # Compute the frequencies


    ax8[0][0].imshow(img)
    ax8[0][0].set_title("Original image")
    ax8[0][1].imshow(magnitude)
    ax8[0][1].set_title("Magnitude")
    ax8[1][0].imshow(phase)
    ax8[1][0].set_title("Phase")
    #ax8[1][1].imshow(np.linspace(freqs.min(), freqs.max(), freqs.size), freqs)
    print("Freqs: ", freqs)
    ax8[1][1].set_title("Frequencies")
    # mangnitude[np.argsort(freqs)]

    threshold = threshold_multiplier * magnitude.max()  # Compute the threshold

    print("Threshold: ", threshold)

    magnitude[magnitude < threshold] = 0  # Apply the threshold (zero out the values below the threshold)

    mean = np.mean(magnitude)
    std = np.std(magnitude)

    magnitude = (magnitude - mean) / std  # Normalize the magnitude

    img_fft = magnitude * np.exp(1j * phase)  # Apply the normalization
    denoised_img = np.real(np.fft.ifft2(img_fft))  # Apply the inverse Fourier transform (we keep only the real part)

    # Normalize image
    denoised_img = np.clip(np.abs(denoised_img) / np.abs(denoised_img).max() * 255, 0, 255).astype(np.uint8)

    return denoised_img


def denoise_blur(img):
    """
    Function that denoises an image using a blur kernel.
    :param img: Image for denoising [NDArray either grayscale or RGB]
    :param kernel_size: Size of the kernel
    :return: Denoised image [NDArray]
    """

    # If the image if float type, we convert it to 8-bit uint8
    if img.max() <= 1:
        img *= 255
        img = img.astype(np.uint8)
        img = np.clip(img, 0, 255).astype(np.uint8)

    denoised_img_full = np.zeros_like(img)

    # If the image is RGB, we denoise each channel separately by recursion
    if len(img.shape) > 2 and img.shape[2] > 1:
        for ch in range(img.shape[2]):
            denoised_img_full[:, :, ch] = denoise_blur(img[:, :, ch])
        return denoised_img_full

    kernel = np.array([[1, 4, 6, 4, 1],
                         [4, 16, 24, 16, 4],
                         [6, 24, 36, 24, 6],
                         [4, 16, 24, 16, 4],
                         [1, 4, 6, 4, 1]], dtype='float64')
    kernel /= np.sum(kernel)

    denoised_img = conv.convolve2d(img, kernel)

    return denoised_img
