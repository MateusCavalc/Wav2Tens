from math import floor
import pyaudio
import wave

from utils import *

FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 4096

def mono_mode(filepath, debug):
    global LOW_FREQ_MAX
    global HIGH_FREQ_MIN

    sampFreq, signal = wavfile.read(filepath)

    print("> Signal dtype:", signal.dtype)
    print("> Signal sample freq:", sampFreq)
    print("> Signal size:", signal.shape[0])

    sampleTime = 1 / sampFreq

    print("> Sample time :", sampleTime, "s")
    length_in_s = signal.shape[0] / sampFreq

    print("> Áudio duration: {:.2f}s".format(length_in_s))

    # Com OVERLAP
    _delay = (int)((FFT_OVERLAP) * FFT_SIZE) * sampleTime * 1000 # millis

    print("> Delay :", _delay, "millis")

    # print("Signal:", signal)

    l_audio = len(signal.shape)
    print ("> Channels:", l_audio)

    if l_audio == 2:
        signal = signal[:,0]

    N = signal.shape[0]

    print("Signal len:", N)

    print("Signal (single channel):", signal)

    # norm
    signal = np.int32(signal)

    # signal = (signal - signal.min()) / (signal.max() - signal.min())
    signal = np.interp(signal, (signal.min(), signal.max()), (-1, 1))

    actual_signal = pos(signal)
    print('> Before normalize => min = {0}; max = {1}'.format(min(actual_signal), max(actual_signal)))

    # Filter requirements.
    fs = sampFreq      # sample rate, Hz
    cutoff = 20      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz  # Nyquist Frequency
    order = 2       # sin wave can be approx represented as quadratic

    # apply filter
    filtered_signal = butter_lowpass_filter(actual_signal, cutoff, fs, order)

    # apply normalization
    filtered_signal = np.interp(filtered_signal, (filtered_signal.min(), filtered_signal.max()), (0, 1))

    print('> After filter and normalize => min = {0}; max = {1}'.format(min(filtered_signal), max(filtered_signal)))

    time_array = np.arange(N) / N * length_in_s

    # -------- FFT --------

    fft_spectrum = np.fft.rfft(signal)
    freq = np.fft.rfftfreq(N, d=1./sampFreq)
    fft_spectrum_abs = np.abs(fft_spectrum)
    # fft_spectrum_abs = np.interp(fft_spectrum_abs, (fft_spectrum_abs.min(), fft_spectrum_abs.max()), (0, 1))
    # fft_spectrum_abs = (fft_spectrum_abs / np.finfo(np.float64).max) * TENS_MAX_MOD

    # ---------------------

    figure, axis = plt.subplots(4, 1)

    axis[0].plot(time_array, signal)
    axis[0].set_title(".wav file")

    axis[1].plot(actual_signal)
    axis[1].set_title("sliced {0} samples".format(ITER_JUMP))

    axis[2].plot(filtered_signal)
    axis[2].set_title("Filtered signal")

    axis[3].plot(freq, fft_spectrum_abs)
    axis[3].set_title("Signal frequency")
    # axis[1, 1].set_xlim([-100, 2000])

    plt.show()

    if debug:
        plt.show(block=False)
        # pass

    # Muda a frequência de corte do filtro
    cutoff = 1000
    order = 2

    tens_low_amp = []
    tens_low_freq = []
    tens_mid_amp = []
    tens_mid_freq = []
    tens_high_amp = []
    tens_high_freq = []

    processedSamples = 0

    print()

    window = 0.5 * (1 - np.cos(np.linspace(0, 2 * np.pi, FFT_SIZE, 1/sampFreq, False)))

    # Pré processamento para pegar pico máximo de amplitude para normalização
    log("Mono", "Iniciando pré processamento", 0)
    max_amp = get_max_amp_from_fft_windows(signal, window, sampFreq)

    # amp_thresh = 0.05 * max_amp
    amp_thresh = 0

    start = timeit.default_timer()

    for fft_index in range(0, len(signal), (int)((FFT_OVERLAP) * FFT_SIZE)):

        fftBulk = signal[fft_index:fft_index + FFT_SIZE]

        if len(fftBulk) < FFT_SIZE:
            fftBulk = np.concatenate([np.zeros(FFT_SIZE - len(fftBulk), dtype=float), fftBulk])
        
        processedSamples = fft_index
        processedPct = (processedSamples / len(signal)) * 100

        print("> | {} samples | {}% overlap | {:.2f}%".format(FFT_SIZE, round((1 - FFT_OVERLAP) * 100), processedPct), end='\r')
        fft_spectrum = np.fft.rfft(fftBulk * window, FFT_SIZE)
        freq = np.fft.rfftfreq(FFT_SIZE, d=1./sampFreq)
        fft_spectrum_abs = np.abs(fft_spectrum)
        # fft_spectrum_abs = fft_spectrum_abs.astype(np.float32)

        # Corta inicio do array (Tira ruído)
        fft_spectrum_abs = fft_spectrum_abs[10:len(fft_spectrum_abs)]
        freq = freq[10:len(freq)]

        # Filter
        n = 5  # the larger n is, the smoother curve will be
        b = [1.0 / n] * n
        a = 1
        filtered_spectrum_abs = filtfilt(b, a, fft_spectrum_abs)
        # filtered_spectrum_abs = butter_lowpass_filter(fft_spectrum_abs, cutoff, fs, order)

        # Get low and high frequency indexes for slicing fft data
        low_freq_idx = get_freq_index(freq, LOW_FREQ_MAX)
        high_freq_idx = get_freq_index(freq, HIGH_FREQ_MIN)

        # Get low, mid and high frequency signal arrays
        low_freq_abs_array = fft_spectrum_abs[0:low_freq_idx]
        mid_freq_abs_array = fft_spectrum_abs[low_freq_idx:high_freq_idx]
        high_freq_abs_array = fft_spectrum_abs[high_freq_idx:]

        # Pega picos
        low_freq_peaks_amp, low_freq_peaks = get_peaks(low_freq_abs_array, freq[0:low_freq_idx], NUM_PEAKS)
        mid_freq_peaks_amp, mid_freq_peaks = get_peaks(mid_freq_abs_array, freq[low_freq_idx:high_freq_idx], NUM_PEAKS)
        high_freq_peaks_amp, high_freq_peaks = get_peaks(high_freq_abs_array, freq[high_freq_idx:], NUM_PEAKS)

        # print(low_freq_peaks, low_freq_peaks_amp)
        # print(mid_freq_peaks, mid_freq_peaks_amp)
        # print(high_freq_peaks, high_freq_peaks_amp)

        # figure, axis = plt.subplots(1, 3)

        # axis[0].plot(low_freq_abs_array, label="mean: {:.2f}".format(mean(low_freq_abs_array)))
        # axis[0].legend(loc="upper right")
        # axis[0].set_title("Low Frequency")
        # # axis[0].set_xlim([0, LOW_FREQ])

        # axis[1].plot(mid_freq_abs_array, label="mean: {:.2f}".format(mean(mid_freq_abs_array)))
        # axis[1].legend(loc="upper right")
        # axis[1].set_title("Middle Frequency")
        # # axis[1].set_xlim([LOW_FREQ, HIGH_FREQ])

        # axis[2].plot(high_freq_abs_array, label="mean: {:.2f}".format(mean(high_freq_abs_array)))
        # axis[2].legend(loc="upper right")
        # axis[2].set_title("High Frequency")
        # # axis[2].set_xlim([HIGH_FREQ, sampFreq/2])

        # figure.title("fft indexes {} to {}".format(fft_index, (fft_index + FFT_SIZE) - 1))

        # print("|- Low freq avg:    {0}".format(mean(low_freq_peaks_amp)))
        # # print("|  |- Peaks {0}".format(low_freq_peaks))
        # print("|- Middle freq avg: {0}".format(mean(mid_freq_peaks_amp)))
        # # print("|  |- Peaks {0}".format(mid_freq_peaks))
        # print("|- High freq avg:   {0}".format(mean(high_freq_peaks_amp),))
        # # print("|  |- Peaks {0}".format(high_freq_peaks))
        # print()

        mean_low_amp = mean(low_freq_peaks_amp)
        mean_mid_amp = mean(mid_freq_peaks_amp)
        mean_high_amp = mean(high_freq_peaks_amp)

        if debug:
            # plt.plot(filtered_signal)
            plt.plot(freq, fft_spectrum_abs)
            plt.plot(freq, filtered_spectrum_abs)
            
            if mean_low_amp > amp_thresh:
                plt.plot(low_freq_peaks, low_freq_peaks_amp, '.')
                plt.plot(mean(low_freq_peaks), mean_low_amp, 'x')
                plt.text(mean(low_freq_peaks) + 0.1, mean_low_amp + 0.1, "{:.2f}\n{:.2f}".format(mean_low_amp, mean(low_freq_peaks)), color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
            
            if mean_mid_amp > amp_thresh:
                plt.plot(mid_freq_peaks, mid_freq_peaks_amp, '.')
                plt.plot(mean(mid_freq_peaks), mean_mid_amp, 'x')
                plt.text(mean(mid_freq_peaks) + 0.1, mean_mid_amp + 0.1, "{:.2f}\n{:.2f}".format(mean_mid_amp, mean(mid_freq_peaks)), color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
            
            if mean_high_amp > amp_thresh:
                plt.plot(high_freq_peaks, high_freq_peaks_amp, '.')
                plt.plot(mean(high_freq_peaks), mean_high_amp, 'x')
                plt.text(mean(high_freq_peaks) + 0.1, mean_high_amp + 0.1, "{:.2f}\n{:.2f}".format(mean_high_amp, mean(high_freq_peaks)), color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
            
            plt.axvline(x=LOW_FREQ_MAX, color='green', ls='--')
            plt.axvline(x=HIGH_FREQ_MIN, color='green', ls='--')
            plt.xlim([0, 3 * HIGH_FREQ_MIN])
            plt.ylim([0, 800])
            plt.ion()
            plt.draw()
            plt.pause(0.01)
            # plt.show()

        if mean_low_amp > amp_thresh:
            tens_low_amp.append(mean_low_amp)
            tens_low_freq.append(mean(low_freq_peaks))
        else:
            tens_low_amp.append(0)
            tens_low_freq.append(0)

        if mean_mid_amp > amp_thresh:
            tens_mid_amp.append(mean_mid_amp)
            tens_mid_freq.append(mean(mid_freq_peaks))
        else:
            tens_mid_amp.append(0)
            tens_mid_freq.append(0)

        if mean_high_amp > amp_thresh:
            tens_high_amp.append(mean_high_amp)            
            tens_high_freq.append(mean(high_freq_peaks))
        else:
            tens_high_amp.append(0)
            tens_high_freq.append(0)

        if debug:
            plt.clf()

    stop = timeit.default_timer()

    print()

    print("[Process duration: {:.2f}s]".format(stop - start))

    # Normaliza array de amplitudes
    tens_low_amp = tens_low_amp / max_amp
    tens_mid_amp = tens_mid_amp / max_amp
    tens_high_amp = tens_high_amp / max_amp

    n = 2 # the larger n is, the smoother curve will be
    b = [1.0 / n] * n
    a = 1

    # Filtra o array de amplitudes
    filtered_low_amp = filtfilt(b, a, tens_low_amp)
    filtered_mid_amp = filtfilt(b, a, tens_mid_amp)
    filtered_high_amp = filtfilt(b, a, tens_high_amp)

    # Filtra o array de frequências
    filtered_low_freq = filtfilt(b, a, tens_low_freq)
    filtered_mid_freq = filtfilt(b, a, tens_mid_freq)
    filtered_high_freq = filtfilt(b, a, tens_high_freq)

    # MOSTRA OS SINAIS DE AMPLITUDE E FREQUÊNCIA
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)

    ax1.plot(tens_low_amp, label='LOW AMP')
    ax1.plot(filtered_low_amp, label='LOW AMP FILTERED')
    ax1.set_title("Low Frequency Amplitude")
    ax2.plot(tens_low_freq, label='LOW FREQ')
    ax2.plot(filtered_low_freq, label='LOW FREQ FILTERED')
    ax2.set_title("Low Frequency")
    # ax2.set_ylim([0, 1])

    ax3.plot(tens_mid_amp, label='MID AMP')
    ax3.plot(filtered_mid_amp, label='MID AMP FILTERED')
    ax3.set_title("Mid Frequency Amplitude")
    ax4.plot(tens_mid_freq, label='MID FREQ')
    ax4.plot(filtered_mid_freq, label='MID FREQ FILTERED')
    ax4.set_title("Mid Frequency")
    # ax4.set_ylim([0, 1])

    ax5.plot(tens_high_amp, label='HIGH AMP')
    ax5.plot(filtered_high_amp, label='HIGH AMP FILTERED')
    ax5.set_title("High Frequency Amplitude")
    ax6.plot(filtered_high_freq, label='HIGH FREQ FILTERED')
    ax6.plot(tens_high_freq, label='HIGH FREQ')
    ax6.set_title("High Frequency")
    # ax6.set_ylim([0, 1])

    plt.show()

    # Atualiza limites de frequência (LIMITES DE FREQUÊNCIA DINÂMICO)
    LOW_FREQ_MIN = min(filtered_low_freq)
    LOW_FREQ_MAX = max(filtered_low_freq)
    MID_FREQ_MIN = min(filtered_mid_freq)
    MID_FREQ_MAX = max(filtered_mid_freq)
    HIGH_FREQ_MIN = min(filtered_high_freq)
    HIGH_FREQ_MAX = max(filtered_high_freq)

    # APLICA TRANSFORMAÇÃO PARA ESCALA
    for i in range(len(filtered_low_freq)):
        filtered_low_amp[i] = floor(filtered_low_amp[i] * TENS_MAX_MOD)

        if filtered_low_freq[i] > 0:
            filtered_low_freq[i] = round(TENS_LOW_FREQ_MIN + (filtered_low_freq[i] - LOW_FREQ_MIN) * (TENS_LOW_FREQ_MAX - TENS_LOW_FREQ_MIN) / (LOW_FREQ_MAX - LOW_FREQ_MIN))
        else:
            filtered_low_freq[i] = TENS_LOW_FREQ_MIN

    for i in range(len(filtered_mid_freq)):
        filtered_mid_amp[i] = floor(filtered_mid_amp[i] * TENS_MAX_MOD)

        if filtered_mid_freq[i] > 0:
            filtered_mid_freq[i] = round(TENS_MID_FREQ_MIN + (filtered_mid_freq[i] - MID_FREQ_MIN) * (TENS_MID_FREQ_MAX - TENS_MID_FREQ_MIN) / (MID_FREQ_MAX - MID_FREQ_MIN))
        else:
            filtered_mid_freq[i] = TENS_MID_FREQ_MIN

    for i in range(len(filtered_high_freq)):
        filtered_high_amp[i] = floor(filtered_high_amp[i] * TENS_MAX_MOD)

        if filtered_high_freq[i] > 0:
            filtered_high_freq[i] = round(TENS_HIGH_FREQ_MIN + (filtered_high_freq[i] - HIGH_FREQ_MIN) * (TENS_HIGH_FREQ_MAX - TENS_HIGH_FREQ_MIN) / (HIGH_FREQ_MAX - HIGH_FREQ_MIN))
        else:
            filtered_high_freq[i] = TENS_HIGH_FREQ_MIN

    # MOSTRA OS SINAIS DE AMPLITUDE E FREQUÊNCIA TRATADOS
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)

    ax1.plot(filtered_low_amp, label='TRANSLATED LOW AMP')
    ax1.set_title("Low Frequency Amplitude")
    ax1.set_ylim([0, TENS_MAX_MOD])
    ax2.plot(filtered_low_freq, label='TRANSLATED LOW FREQ')
    ax2.set_title("Low Frequency")
    # ax2.set_ylim([0, 1])

    ax3.plot(filtered_mid_amp, label='TRANSLATED MID AMP')
    ax3.set_title("Mid Frequency Amplitude")
    ax3.set_ylim([0, TENS_MAX_MOD])
    ax4.plot(filtered_mid_freq, label='TRANSLATED MID FREQ')
    ax4.set_title("Mid Frequency")
    # ax4.set_ylim([0, 1])

    ax5.plot(filtered_high_amp, label='TRANSLATED HIGH AMP')
    ax5.set_title("High Frequency Amplitude")
    ax5.set_ylim([0, TENS_MAX_MOD])
    ax6.plot(filtered_high_freq, label='TRANSLATED HIGH FREQ')
    ax6.set_title("High Frequency")
    # ax6.set_ylim([0, 1])

    plt.show()

    print("[Starting transmition ...]")

    # sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # server_address = ('192.168.4.1', 5000)
    # print('\n[Connecting to {0}] ...'.format(server_address))
    # sock.connect(server_address)
    # print('[Connected]\n')
    # sock.sendall(b"command")
    # time.sleep(1000 / 1000)

    playsound(filepath, block=False)
    time.sleep(200 / 1000)

    start = timeit.default_timer()

    for i in range(len(filtered_low_amp)):
        print("|- Channel 1 (LOW): {0}".format((filtered_low_amp[i], filtered_low_freq[i])))
        print("|- Channel 2 (MID): {0}".format((filtered_mid_amp[i], filtered_mid_freq[i])))
        print("|- Channel 3 (HIGH): {0}".format((filtered_high_amp[i], filtered_high_freq[i])))
        print()

        channel1_string = '1.{0}.{1}'.format(filtered_low_amp[i], filtered_low_freq[i])
        channel2_string = '1.{0}.{1}'.format(filtered_mid_amp[i], filtered_mid_freq[i])
        channel3_string = '1.{0}.{1}'.format(filtered_high_amp[i], filtered_high_freq[i])

        tens_packet = channel1_string + ';' + channel2_string + ';' + channel3_string

        # sock.sendall(tens_packet.encode('utf-8'))
        time.sleep(_delay / 1000)

    stop = timeit.default_timer()

    print("[Transmition duration: {:.2f}s]".format(stop - start))
    print("[File duration: {:.2f}s]".format(length_in_s))

    # time.sleep(100 / 1000)
    # sock.sendall(b"quit")
    # time.sleep(1000 / 1000)
    # sock.close()

def get_max_amp_from_fft_windows(signal, window, sampFreq):

    #               [L, M, H]
    file_max_amps = [0, 0, 0]

    for fft_index in range(0, len(signal), (int)((FFT_OVERLAP) * FFT_SIZE)):

        fftBulk = signal[fft_index:fft_index + FFT_SIZE]

        if len(fftBulk) < FFT_SIZE:
            fftBulk = np.concatenate([np.zeros(FFT_SIZE - len(fftBulk), dtype=float), fftBulk])
        
        processedSamples = fft_index
        processedPct = (processedSamples / len(signal)) * 100

        fft_spectrum = np.fft.rfft(fftBulk * window, FFT_SIZE)
        freq = np.fft.rfftfreq(FFT_SIZE, d=1./sampFreq)
        fft_spectrum_abs = np.abs(fft_spectrum)
        # fft_spectrum_abs = fft_spectrum_abs.astype(np.float32)

        # Get low and high frequency indexes for slicing fft data
        low_freq_idx = get_freq_index(freq, LOW_FREQ_MAX)
        high_freq_idx = get_freq_index(freq, HIGH_FREQ_MIN)

        # Get low, mid and high frequency signal arrays
        low_freq_abs_array = fft_spectrum_abs[0:low_freq_idx]
        mid_freq_abs_array = fft_spectrum_abs[low_freq_idx:high_freq_idx]
        high_freq_abs_array = fft_spectrum_abs[high_freq_idx:]

        window_max_low = max(low_freq_abs_array)      
        window_max_mid = max(mid_freq_abs_array)      
        window_max_high = max(high_freq_abs_array)      

        if window_max_low > file_max_amps[0]:
            file_max_amps[0] = window_max_low

        if window_max_mid > file_max_amps[1]:
            file_max_amps[1] = window_max_mid

        if window_max_high > file_max_amps[2]:
            file_max_amps[2] = window_max_high

    return max(file_max_amps)

def mono_mode_realtime(debug):
    global LOW_FREQ_MAX
    global HIGH_FREQ_MIN

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

    frames = []
    fftBulk = []

    plt.ion()
    fig = plt.figure()
    canva = fig.add_subplot(111)
    signal_plot, = canva.plot(np.linspace(0, CHUNK, CHUNK, endpoint=False), np.zeros(4096, dtype=np.int32))

    while True:
        frames = stream.read(CHUNK)
        print("> Frames:", type(frames), len(frames))
    
        for index in range(0, len(frames), 4):
            sample = int.from_bytes(frames[index:index + 3], byteorder='little')
            fftBulk.append(sample)

        print("> fftBulk infos:", type(fftBulk), len(fftBulk))
        # print("> fftBulk:", fftBulk)

        # Muda a frequência de corte do filtro
        cutoff = 1000
        order = 2
        
        # print(fftBulk)
        fft_spectrum = np.fft.rfft(fftBulk, CHUNK)
        # print(fft_spectrum)
        freq = np.fft.rfftfreq(CHUNK, d=1./RATE)
        fft_spectrum_abs = np.abs(fft_spectrum)
        # fft_spectrum_abs = fft_spectrum_abs.astype(np.float32)

        # Corta inicio do array (Tira ruído)
        fft_spectrum_abs = fft_spectrum_abs[10:len(fft_spectrum_abs)]
        freq = freq[10:len(freq)]

        # Filter
        n = 5  # the larger n is, the smoother curve will be
        b = [1.0 / n] * n
        a = 1
        filtered_spectrum_abs = filtfilt(b, a, fft_spectrum_abs)
        # filtered_spectrum_abs = butter_lowpass_filter(fft_spectrum_abs, cutoff, fs, order)

        # Get low and high frequency indexes for slicing fft data
        low_freq_idx = get_freq_index(freq, LOW_FREQ_MAX)
        high_freq_idx = get_freq_index(freq, HIGH_FREQ_MIN)

        # Get low, mid and high frequency signal arrays
        low_freq_abs_array = fft_spectrum_abs[0:low_freq_idx]
        mid_freq_abs_array = fft_spectrum_abs[low_freq_idx:high_freq_idx]
        high_freq_abs_array = fft_spectrum_abs[high_freq_idx:]

        # Pega picos
        low_freq_peaks_amp, low_freq_peaks = get_peaks(low_freq_abs_array, freq[0:low_freq_idx], NUM_PEAKS)
        mid_freq_peaks_amp, mid_freq_peaks = get_peaks(mid_freq_abs_array, freq[low_freq_idx:high_freq_idx], NUM_PEAKS)
        high_freq_peaks_amp, high_freq_peaks = get_peaks(high_freq_abs_array, freq[high_freq_idx:], NUM_PEAKS)

        # print(low_freq_peaks, low_freq_peaks_amp)
        # print(mid_freq_peaks, mid_freq_peaks_amp)
        # print(high_freq_peaks, high_freq_peaks_amp)

        # figure, axis = plt.subplots(1, 3)

        # axis[0].plot(low_freq_abs_array, label="mean: {:.2f}".format(mean(low_freq_abs_array)))
        # axis[0].legend(loc="upper right")
        # axis[0].set_title("Low Frequency")
        # # axis[0].set_xlim([0, LOW_FREQ])

        # axis[1].plot(mid_freq_abs_array, label="mean: {:.2f}".format(mean(mid_freq_abs_array)))
        # axis[1].legend(loc="upper right")
        # axis[1].set_title("Middle Frequency")
        # # axis[1].set_xlim([LOW_FREQ, HIGH_FREQ])

        # axis[2].plot(high_freq_abs_array, label="mean: {:.2f}".format(mean(high_freq_abs_array)))
        # axis[2].legend(loc="upper right")
        # axis[2].set_title("High Frequency")
        # # axis[2].set_xlim([HIGH_FREQ, sampFreq/2])

        # figure.title("fft indexes {} to {}".format(fft_index, (fft_index + FFT_SIZE) - 1))

        # print("|- Low freq avg:    {0}".format(mean(low_freq_peaks_amp)))
        # # print("|  |- Peaks {0}".format(low_freq_peaks))
        # print("|- Middle freq avg: {0}".format(mean(mid_freq_peaks_amp)))
        # # print("|  |- Peaks {0}".format(mid_freq_peaks))
        # print("|- High freq avg:   {0}".format(mean(high_freq_peaks_amp),))
        # # print("|  |- Peaks {0}".format(high_freq_peaks))
        # print()

        mean_low_amp = mean(low_freq_peaks_amp)
        mean_mid_amp = mean(mid_freq_peaks_amp)
        mean_high_amp = mean(high_freq_peaks_amp)

        if debug:
            # plt.plot(filtered_signal)
            # plt.plot(fftBulk)
            # plt.plot(freq, fft_spectrum_abs)
            # plt.plot(freq, filtered_spectrum_abs)
            
            # plt.plot(low_freq_peaks, low_freq_peaks_amp, '.')
            # plt.plot(mean(low_freq_peaks), mean_low_amp, 'x')
            # plt.text(mean(low_freq_peaks) + 0.1, mean_low_amp + 0.1, "{:.2f}\n{:.2f}".format(mean_low_amp, mean(low_freq_peaks)), color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
        
            # plt.plot(mid_freq_peaks, mid_freq_peaks_amp, '.')
            # plt.plot(mean(mid_freq_peaks), mean_mid_amp, 'x')
            # plt.text(mean(mid_freq_peaks) + 0.1, mean_mid_amp + 0.1, "{:.2f}\n{:.2f}".format(mean_mid_amp, mean(mid_freq_peaks)), color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
        
            # plt.plot(high_freq_peaks, high_freq_peaks_amp, '.')
            # plt.plot(mean(high_freq_peaks), mean_high_amp, 'x')
            # plt.text(mean(high_freq_peaks) + 0.1, mean_high_amp + 0.1, "{:.2f}\n{:.2f}".format(mean_high_amp, mean(high_freq_peaks)), color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
            
            # plt.axvline(x=LOW_FREQ_MAX, color='green', ls='--')
            # plt.axvline(x=HIGH_FREQ_MIN, color='green', ls='--')
            # plt.xlim([0, 3 * HIGH_FREQ_MIN])
            # plt.ylim([0, 800])
            signal_plot.set_ydata(fftBulk)
            fig.canvas.draw()
            fig.canvas.flush_events()
            # plt.show()

        if debug:
            # plt.clf()
            pass

        fftBulk.clear()
    
    stream.stop_stream()
    stream.close()
    p.terminate()