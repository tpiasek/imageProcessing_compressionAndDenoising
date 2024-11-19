import numpy as np
import pywt
import matplotlib.pyplot as plt


def compress_wavelet_img(img, compression_rate, wavelet_name="db1", level=3):
    """
    Function that compresses an image using the wavelet transform.

    :param img: image for compression (can be either grayscale or RGB)
    :param compression_rate: how much we want to compress the image (between 0 and 1)
    :param wavelet_name: name of the wavelet to use
    :param level: level of the wavelet transform
    :return: tuple(compressed image, frequency spectrum)
    """

    # If the image is boolean type, we convert it to uint8
    if np.asarray(img).dtype == np.bool_:
        img = img.astype(np.uint8)

    # If the image is float type, we convert it to uint8
    if img.max() <= 1:
        img *= 255
        img = img.astype(np.uint8)
        img = np.clip(img, 0, 255)

    # If the image is RGB, we compress each channel separately
    if len(img.shape) > 2 and img.shape[2] > 1:
        spectrum = np.zeros(img.shape)
        comp_img = np.zeros(img.shape)
        for ch in range(img.shape[2]):
            # We compress each channel separately by calling the function recursively
            comp_img[:, :, ch], spectrum_raw = compress_wavelet_img(img[:, :, ch], compression_rate, wavelet_name, level)
            # Because sometimes the wavelet transform returns an image with one more column than the original image, we remove the first column
            if spectrum_raw.shape != spectrum[:, :, ch].shape:
                spectrum[:, :, ch] = spectrum_raw[:, 1:]
            else:
                spectrum[:, :, ch] = spectrum_raw
        # We normalize the image
        comp_img = np.asarray(np.clip(comp_img / comp_img.max() * 255, 0, 255), dtype=np.uint8)
        return comp_img, np.clip(spectrum / spectrum.max() * 255, 0, 255).astype(np.uint8)

    img_wav = pywt.wavedec2(img, wavelet_name, level=level)  # Apply wavelet transform to the image
    approx = img_wav[0]
    details = img_wav[1:]
    transformed, slices = pywt.coeffs_to_array(img_wav)  # Flatten the image

    coeff = np.sort(np.abs(transformed.reshape(-1)))  # Sort the image
    threshold = coeff[int(compression_rate * len(coeff))]  # Compute the threshold
    indices = np.abs(transformed) > threshold  # Check which values are above the threshold (those are the ones we keep)

    decompressed = transformed * indices  # Apply the compression
    # spectrum = np.log(1 + np.abs(np.fft.fftshift(decompressed)))  # Compute the frequency spectrum
    spectrum = np.log(1 + np.abs(decompressed))  # Compute the frequency spectrum
    spectrum = np.clip(spectrum / spectrum.max() * 255, 0, 255).astype(np.uint8)

    final_img = pywt.waverec2(pywt.array_to_coeffs(decompressed, slices, output_format="wavedec2"), wavelet_name)  # Apply the inverse wavelet transform

    return final_img, spectrum


def compress_fourier_img(img, compression_rate):
    """
    Function that compresses an image using the Fourier transform.

    :param img: image for compression (can be either grayscale or RGB)
    :param compression_rate: how much we want to compress the image (between 0 and 1)
    :return: tuple(compressed image, frequency spectrum)
    """

    # If the image is boolean type, we convert it to uint8
    if np.asarray(img).dtype == np.bool_:
        img = img.astype(np.uint8)

    # If the image is float type, we convert it to uint8
    if img.max() <= 1:
        img *= 255
        img = img.astype(np.uint8)
        img = np.clip(img, 0, 255)

    # If the image is RGB, we compress each channel separately
    if len(img.shape) > 2 and img.shape[2] > 1:
        spectrum = np.zeros(img.shape)
        comp_img = np.zeros(img.shape)
        for ch in range(img.shape[2]):
            # We compress each channel separately by calling the function recursively
            comp_img[:, :, ch], spectrum[:, :, ch] = compress_fourier_img(img[:, :, ch], compression_rate)
        # We normalize the image
        comp_img = np.asarray(np.clip(comp_img / comp_img.max() * 255, 0, 255), dtype=np.uint8)
        return comp_img.astype(np.uint8), spectrum

    img_fft = np.fft.fft2(img)  # Apply Fourier transform to the image
    img_fft_flattened = np.abs(img_fft.reshape(-1))  # Flatten the image
    coefficients = np.sort(img_fft_flattened)  # Sort the image

    threshold = coefficients[int(compression_rate * len(coefficients))]
    indices = np.abs(img_fft) > threshold  # Check which values are above the threshold (those are the ones we keep)

    img_fft_compressed = img_fft * indices  # Apply the compression
    spectrum = np.log(1 + np.abs(np.fft.fftshift(img_fft_compressed))) # Compute the frequency spectrum
    spectrum = np.clip(spectrum, 0, 1)
    img_fft_decompressed = np.fft.ifft2(img_fft_compressed)  # Apply the inverse Fourier transform

    print_size_data(img, img_fft_compressed)

    return np.abs(img_fft_decompressed), spectrum


def print_size_data(img, compressed_img):
    original_size = img.size
    compressed_size = np.count_nonzero(compressed_img)
    size_reduction = 1 - (compressed_size / original_size)

    print(f"Original size: {original_size}")
    print(f"Compressed size: {compressed_size}")
    print(f"Size reduction: {size_reduction} \n")

    return size_reduction
