import numpy as np
import config
from scipy import signal

np.random.seed(45)


bands = {'delta': [0.5, 4],
         'theta': [4, 8],
         'alpha': [8, 13],
         'beta': [13, 30],
         'gamma': [30, 100]}

# Thanks to ChatGPT and Github :)

def random_crop(eeg, **kwargs):
    """
    Randomly crops the input EEG signal.
    Args:
        eeg: Input EEG signal of shape (n_channels, n_samples)
        crop_size: Tuple containing the size of the crop in (channels, samples)
    Returns:
        Cropped EEG signal of shape (n_channels, crop_size[1])
    """
    n_channels, n_samples = eeg.shape
    crop_channels, crop_samples = kwargs['crop_size']

    # Check if crop size is smaller than input size
    if crop_channels > n_channels or crop_samples > n_samples:
        raise ValueError("Crop size should be smaller than input size.")

    # Generate random indices for cropping
    crop_channel_idx = np.random.randint(0, n_channels - crop_channels + 1)
    crop_sample_idx = np.random.randint(0, n_samples - crop_samples + 1)

    # Crop the input signal
    cropped_eeg = eeg[crop_channel_idx:crop_channel_idx + crop_channels, crop_sample_idx:crop_sample_idx + crop_samples]

    pad_size    = n_samples - cropped_eeg.shape[1]
    cropped_eeg = np.pad(cropped_eeg, ((0, 0), (pad_size, 0)), mode="constant")

    return cropped_eeg


def time_shift(eeg, **kwargs):
    """
    Shifts the input EEG signal in time.
    Args:
        eeg: Input EEG signal of shape (n_channels, n_samples)
        shift: Number of samples to shift the signal (positive or negative)
    Returns:
        Shifted EEG signal of shape (n_channels, n_samples)
    """
    if kwargs['max_shift'] == 0:
        return eeg

    shift = 0
    while shift==0:
        shift = np.random.randint(-kwargs['max_shift'], kwargs['max_shift'])

    n_channels, n_samples = eeg.shape

    # Check if shift is within the signal length
    if abs(shift) > n_samples:
        raise ValueError("Shift should be smaller than signal length.")

    # Pad the signal with zeros for the shifted samples
    if shift > 0:
        shifted_eeg = np.pad(eeg[:, shift:], ((0, 0), (0, shift)), mode="constant")
    else:
        shifted_eeg = np.pad(eeg[:, :shift], ((0, 0), (abs(shift), 0)), mode="constant")

    return shifted_eeg


def channel_shuffle(eeg, **kwargs):
    """
    Shuffles the order of channels in the input EEG signal.
    Args:
        eeg: Input EEG signal of shape (n_channels, n_samples)
    Returns:
        Shuffled EEG signal of shape (n_channels, n_samples)
    """
    n_channels, n_samples = eeg.shape

    # Generate a random permutation of channel indices
    permuted_indices = np.random.permutation(n_channels)

    # Shuffle the channels according to the permutation
    shuffled_eeg = eeg[permuted_indices, :]

    return shuffled_eeg


def random_noise(eeg, **kwargs):
    noise = np.random.randn(*eeg.shape) * kwargs['noise_factor']
    return eeg + noise


def apply_augmentation(signal, aug_type, **kwargs):
    if aug_type == 'time_shift':
        return time_shift(signal, **kwargs)
    elif aug_type == 'channel_shuffle':
        return channel_shuffle(signal, **kwargs)
    elif aug_type == 'random_crop':
        return random_crop(signal, **kwargs)
    elif aug_type == 'random_noise':
        return random_noise(signal, **kwargs)
    elif aug_type == 'all':
        # return random_noise(channel_shuffle(time_shift(random_crop(signal, **kwargs), **kwargs), **kwargs), **kwargs)
        return random_noise(random_crop(signal, **kwargs), **kwargs)
    else:
        raise ValueError("Invalid augmentation type")

def extract_band(power, freq, band):
    idx = np.logical_and(freq >= band[0], freq < band[1])
    return np.trapz(power[:, idx], axis=1)

def extract_freq_band(eeg, fs=1000, nperseg=440):
    f, psd = signal.welch(eeg.T, fs=fs, nperseg=nperseg, axis=1)
    # delta  = extract_band(psd, f, bands['delta'])
    # theta  = extract_band(psd, f, bands['theta'])
    # alpha  = extract_band(psd, f, bands['alpha'])
    # beta   = extract_band(psd, f, bands['beta'])
    gamma  = extract_band(psd, f, bands['gamma'])
    return gamma

if __name__ == '__main__':
    # Load EEG signal from file or generate it
    signal = np.random.rand(128, 440)

    # # Apply time shift augmentation with max shift of 10 samples
    # aug_signal = apply_augmentation(signal, 'time_shift', max_shift=10)
    # print(aug_signal.shape, 'mse error: {}'.format( np.mean((signal-aug_signal)**2) ))

    # # Apply random crop augmentation with crop size of 400 samples
    # aug_signal = apply_augmentation(signal, 'random_crop', crop_size=(128, 380))
    # print(aug_signal.shape, 'mse error: {}'.format( np.mean((signal-aug_signal)**2) ))

    # # Apply random noise augmentation with noise factor of 0.1
    # aug_signal = apply_augmentation(signal, 'random_noise', noise_factor=0.2)
    # print(aug_signal.shape, 'mse error: {}'.format( np.mean((signal-aug_signal)**2) ))

    # # Apply random channel shuffle
    # aug_signal = apply_augmentation(signal, 'channel_shuffle')
    # print(aug_signal.shape, 'mse error: {}'.format( np.mean((signal-aug_signal)**2) ))
    aug_sginal= apply_augmentation(signal, 'all', max_shift=config.max_shift, crop_size=config.crop_size, noise_factor=config.noise_factor)
    print(aug_sginal.shape)