import librosa
import numpy
import scipy
def mfcc(data,sr=16000,window_len=1024,hop_len=512,n_fft=1024,n_mfcc=13,n_mel=40):
    """Static MFCC

    Parameters
    ----------
    data : numpy.ndarray
        Audio data
    params : dict
        Parameters

    Returns
    -------
    list of numpy.ndarray
        List of feature matrices, feature matrix per audio channel

    """

    window = scipy.signal.hamming(window_len, sym=False)

    mel_basis = librosa.filters.mel(sr=sr,
                                    n_fft=n_fft,
                                    n_mels=n_mel,
                                    htk=False)

    # if params.get('normalize_mel_bands'):
    #     mel_basis /= numpy.max(mel_basis, axis=-1)[:, None]

    feature_matrix = []
    spectrogram_ = numpy.abs(librosa.stft(data,
                            n_fft=n_fft,
                            win_length=window_len,
                            hop_length=hop_len,
                            center=True))
    mel_spectrum = numpy.dot(mel_basis, spectrogram_)

    mfcc = librosa.feature.mfcc(S=librosa.logamplitude(mel_spectrum),
                                    n_mfcc=n_mfcc)

    feature_matrix.append(mfcc.T)

    return mfcc