import threading
from math import floor

from utils import *

tens_amps = [[], [], []]
filtered_tens_amps = [[], [], []]
tens_freqs = [[], [], []]
filtered_tens_freqs = [[], [], []]

OLD_FREQ_SCALE_MAXS = []
OLD_FREQ_SCALE_MINS = []

subcommands = [[], [], []]
commands = []

_delays = []
_audio_lens = []

# NOVA ESCALA SENSORIAL TÁTIL
NEW_FREQ_SCALE_MIN = 30
NEW_FREQ_SCALE_MAX = 80
pace = 1

def many_mode(filepaths, debug):

    threads = list()

    for index, filepath in enumerate(filepaths):
        x = threading.Thread(target=process_channel, args=("canal_{0}".format(index+1), filepath, tens_amps[index], tens_freqs[index], debug))
        threads.append(x)
        x.start()

    for index, thread in enumerate(threads):
        print("Main : Joining thread {}.".format(index))
        thread.join()

    # Corta os arrays para o menor tamanho entre eles
    min_size = len(tens_amps[0])

    for channel_amps in tens_amps:
        if len(channel_amps) > 0:
            min_size = min(min_size, len(channel_amps))

    for i in range(len(tens_amps)):
        tens_amps[i] = tens_amps[i][0:min_size]
        tens_freqs[i] = tens_freqs[i][0:min_size]

    filtered_tens_amps[0] = tens_amps[0]
    filtered_tens_amps[1] = tens_amps[1]
    filtered_tens_amps[2] = tens_amps[2]
    filtered_tens_freqs[0] = tens_freqs[0]
    filtered_tens_freqs[1] = tens_freqs[1]
    filtered_tens_freqs[2] = tens_freqs[2]

    # Aplica ultimos filtros na AMPLITUDE
    n = 3  # the larger n is, the smoother curve will be
    b = [1.0 / n] * n
    a = 1
    filter_iter = 2
    for iter in range(filter_iter):
        for i in range(len(filtered_tens_amps)):
            if len(filtered_tens_amps[i]) > 0:  
                filtered_tens_amps[i] = filtfilt(b, a, filtered_tens_amps[i])

    # Aplica ultimos filtros na FREQUÊNCIA
    n = 8  # the larger n is, the smoother curve will be
    b = [1.0 / n] * n
    a = 1
    filter_iter = 1
    for iter in range(filter_iter):
        for i in range(len(filtered_tens_freqs)):
            if len(filtered_tens_freqs[i]) > 0:   
                filtered_tens_freqs[i] = filtfilt(b, a, filtered_tens_freqs[i])

    # MOSTRA OS PICOS DE AMPLITUDE
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(tens_amps[0], label="Channel 1")
    ax1.plot(tens_amps[1], label="Channel 2")
    ax1.plot(tens_amps[2], label="Channel 3")
    # ax1.plot(filtered_tens_amps[0], label="F_Channel 1")
    # ax1.plot(filtered_tens_amps[1], label="F_Channel 2")
    # ax1.plot(filtered_tens_amps[2], label="F_Channel 3")
    ax1.set_title("Channels Amplitudes")
    ax1.legend(loc="upper right")
    ax1.set_xlabel('Samples')
    ax1.set_ylabel('Normalized amplitude')
    ax1.set_ylim([0, 1])
    ax2.plot(tens_freqs[0], label="Channel 1")
    ax2.plot(tens_freqs[1], label="Channel 2")
    ax2.plot(tens_freqs[2], label="Channel 3")
    # ax2.plot(filtered_tens_freqs[0], label="F_Channel 1")
    # ax2.plot(filtered_tens_freqs[1], label="F_Channel 2")
    # ax2.plot(filtered_tens_freqs[2], label="F_Channel 3")
    ax2.set_title("Channels Frequencies")
    ax2.legend(loc="upper right")
    ax2.set_xlabel('Samples')
    ax2.set_ylabel('Frequency')

    plt.show()

    # LIMITES DE FREQUÊNCIA DINÂMICO
    for processed_freqs in tens_freqs:
        if len(processed_freqs) > 0:
            OLD_FREQ_SCALE_MINS.append(min(processed_freqs))
            OLD_FREQ_SCALE_MAXS.append(max(processed_freqs))

    print("> ALL CHANNEL FREQ MINS:", OLD_FREQ_SCALE_MINS)
    print("> ALL CHANNEL FREQ MAXS:", OLD_FREQ_SCALE_MAXS)

    OLD_FREQ_SCALE_MIN = min(OLD_FREQ_SCALE_MINS)
    OLD_FREQ_SCALE_MAX = max(OLD_FREQ_SCALE_MAXS)

    print("---------------------------------")
    print("> DINAMIC MIN FREQ:", OLD_FREQ_SCALE_MIN)
    print("> DINAMIC MAX FREQ:", OLD_FREQ_SCALE_MAX)
    print("---------------------------------")

    # APLICA TRANSFORMAÇÃO PARA ESCALA
    for channel_amps, channel_freqs in zip(tens_amps, tens_freqs):
        channel_amps_len = len(channel_amps)

        if channel_amps_len > 0:
            for i in range(channel_amps_len):
                channel_amps[i] = floor(channel_amps[i] * TENS_MAX_MOD)

                if channel_freqs[i] > 0:
                    channel_freqs[i] = round(NEW_FREQ_SCALE_MIN + (channel_freqs[i] - OLD_FREQ_SCALE_MIN) * (NEW_FREQ_SCALE_MAX - NEW_FREQ_SCALE_MIN) / (OLD_FREQ_SCALE_MAX - OLD_FREQ_SCALE_MIN)) * pace
                else:
                    channel_freqs[i] = NEW_FREQ_SCALE_MIN

    # MOSTRA AS AMPLITUDES E FREQUÊNCIAS TRADUZIDAS
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(tens_amps[0], label="Channel 1")
    ax1.plot(tens_amps[1], label="Channel 2")
    ax1.plot(tens_amps[2], label="Channel 3")
    ax1.set_title("Channels Translated Amplitudes")
    ax1.legend(loc="upper right")
    ax1.set_xlabel('Samples')
    ax1.set_ylabel('Duty cycle amplitude')
    ax1.set_ylim([0, TENS_MAX_MOD])
    ax2.plot(tens_freqs[0], label="Channel 1")
    ax2.plot(tens_freqs[1], label="Channel 2")
    ax2.plot(tens_freqs[2], label="Channel 3")
    ax2.set_title("Channels Translated Frequencies")
    ax2.legend(loc="upper right")
    ax2.set_xlabel('Samples')
    ax2.set_ylabel('Translated frequency')

    plt.show()

    # MONTA SUBCOMANDOS
    for channel_amps, channel_freqs, channel_subcommand in zip(tens_amps, tens_freqs, subcommands):
        channel_amps_len = len(channel_amps)

        if channel_amps_len > 0:
            # Parse as int array
            channel_amps = [int(i) for i in channel_amps]
            channel_freqs = [int(i) for i in channel_freqs]

            for i in range(channel_amps_len):
                c_mod = channel_amps[i]
                c_freq = channel_freqs[i]

                subcommand = '1.{:02d}.{:02d}'.format(c_mod, c_freq)

                channel_subcommand.append(subcommand)

    # # MONTA COMANDOS
    # for i in range(len(subcommands[0])):
    #     command = subcommands[0] + 

    print("[Starting transmition ...]")
    print("> Delays:", _delays)

    _delay = mean(_delays)

    repeat = True

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('192.168.4.1', 5000)
    print('\n[Connecting to {0}] ...'.format(server_address))
    sock.connect(server_address)
    print('[Connected]\n')
    sock.sendall(b"command")
    time.sleep(1000 / 1000)

    # Repete até sair do programa
    while repeat:

        for sound in filepaths:
            playsound(sound, block=False)

        time.sleep(250 / 1000)

        start = timeit.default_timer()

        for i in range(len(subcommands[0])):

            tens_packet = subcommands[0][i] + ';'

            if len(subcommands[1]) > 0:
                tens_packet += subcommands[1][i] + ';'
            else:
                tens_packet += '0.00.01;'

            if len(subcommands[2]) > 0:
                tens_packet += subcommands[2][i]
            else:
                tens_packet += '0.00.01'

            sock.sendall(tens_packet.encode('utf-8'))
            print(" < Sent '{}' >".format(tens_packet), end='\r')

            time.sleep(_delay / 1000)

        stop = timeit.default_timer()
        
        time.sleep(500 / 1000)
        sock.sendall("1.0.1;1.0.1;1.0.1".encode('utf-8'))
        time.sleep(500 / 1000)

        print("[Transmition duration: {:.2f}s]".format(stop - start))
        print("[File durations : {} s]".format(_audio_lens))

        inp = input(" Repeat? (y/n)\n\n > ")
        
        if inp == 'n':
            repeat = False

    time.sleep(100 / 1000)
    sock.sendall(b"quit")
    time.sleep(1000 / 1000)
    sock.close()
    

def process_channel(threadName, filepath, tens_amp, tens_freq, debug):
    print("[{}] - Started processing '{}'.".format(threadName, filepath))

    sampFreq, signal = wavfile.read(filepath)

    print("> Signal dtype:", signal.dtype)
    print("> Signal sample freq:", sampFreq)
    print("> Signal size:", signal.shape[0])

    sampleTime = 1 / sampFreq

    print("> Sample time :", sampleTime, "s")
    length_in_s = signal.shape[0] / sampFreq

    print("> Áudio duration: {:.2f}s".format(length_in_s))
    _audio_lens.append(length_in_s)

    # Com OVERLAP
    _delay = round(((FFT_OVERLAP) * FFT_SIZE) * sampleTime * 1000) # millis

    print("> Delay :", _delay, "millis")
    _delays.append(_delay)

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

    # # figure, axis = plt.subplots(4, 1)

    # # axis[0].plot(time_array, signal)
    # # axis[0].set_title(".wav file")

    # # axis[1].plot(actual_signal)
    # # axis[1].set_title("sliced {0} samples".format(ITER_JUMP))

    # # axis[2].plot(filtered_signal)
    # # axis[2].set_title("Filtered signal")

    # # axis[3].plot(freq, fft_spectrum_abs)
    # # axis[3].set_title("Signal frequency")
    # axis[1, 1].set_xlim([-100, 2000])

    # # plt.show()

    if debug:
        plt.show(block=False)
        # pass

    # Muda a frequência de corte do filtro
    cutoff = 1000
    order = 2

    windows_amps = []
    windows_freqs = []

    processedSamples = 0

    print()

    window = 0.5 * (1 - np.cos(np.linspace(0, 2 * np.pi, FFT_SIZE, 1/sampFreq, False)))

    # Pré processamento para pegar pico máximo de amplitude para normalização
    log(threadName, "Iniciando pré processamento", 0)
    max_amp = get_max_amp_from_fft_windows(signal, window, sampFreq)

    amp_thresh = 0.05 * max_amp
    # amp_thresh = 0

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

        # Corta inicio do array (Tira "frequência 0")
        fft_spectrum_abs = fft_spectrum_abs[2:len(fft_spectrum_abs)]
        freq = freq[2:len(freq)]
        
        # Filter
        n = 5  # the larger n is, the smoother curve will be
        b = [1.0 / n] * n
        a = 1
        filtered_spectrum_abs = filtfilt(b, a, fft_spectrum_abs)
        # filtered_spectrum_abs = butter_lowpass_filter(fft_spectrum_abs, cutoff, fs, order)

        # Pega picos
        freq_peaks_amp, freq_peaks = get_peaks(filtered_spectrum_abs, freq, NUM_PEAKS)

        # print(freq_peaks_amp, freq_peaks)

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

        mean_amp = mean(freq_peaks_amp)

        if debug:
            # plt.plot(filtered_signal)
            plt.plot(freq, fft_spectrum_abs)
            plt.plot(freq, filtered_spectrum_abs)
            
            if mean_amp > amp_thresh:
                plt.plot(freq_peaks, freq_peaks_amp, '.')
                plt.plot(mean(freq_peaks), mean_amp, 'o')
                plt.text(mean(freq_peaks) + 100, mean_amp + 30, "{:.2f}\n{:.2f}".format(mean_amp, mean(freq_peaks)), color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
            
            # plt.axvline(x=LOW_FREQ, color='green', ls='--')
            # plt.axvline(x=HIGH_FREQ, color='green', ls='--')
            plt.xlim([0, 2 * HIGH_FREQ_MIN])
            plt.ylim([0, max_amp])
            plt.ion()
            plt.draw()
            plt.pause(0.01)
            # plt.show()

        if mean_amp > amp_thresh:
            # index = len(windows_freqs) - 1
            # if len(windows_freqs) >= 2:
            #     freqToAppend = (windows_freqs[index] + windows_freqs[index - 1] + mean(freq_peaks)) / 3
            # else:
            #     freqToAppend = mean(freq_peaks)
        
            freqToAppend = mean(freq_peaks)

            windows_amps.append(mean_amp)
            windows_freqs.append(freqToAppend)
        else:
            windows_amps.append(0)
            windows_freqs.append(0)

        if debug:
            plt.clf()

    stop = timeit.default_timer()

    print()

    print("[Process duration: {:.2f}s]".format(stop - start))

    # Normaliza array de amplitudes
    windows_amps = windows_amps / max_amp

    # Filtra o array de amplitudes
    n = 5  # the larger n is, the smoother curve will be
    b = [1.0 / n] * n
    a = 1
    filtered_windows_amps = windows_amps

    # filtered_windows_amps = filtfilt(b, a, windows_amps)

    # Filtra o array de frequências
    n = 5  # the larger n is, the smoother curve will be
    b = [1.0 / n] * n
    filtered_windows_freqs = windows_freqs

    # Aplica uma sequencia de filtros
    # for i in range(20):
    #     filtered_windows_freqs = filtfilt(b, a, filtered_windows_freqs)

    # Valores de amplitude filtrados
    for fw_amp in filtered_windows_amps:
        tens_amp.append(fw_amp)

    # Valores de frequência filtrados
    for fw_freq in filtered_windows_freqs:
        tens_freq.append(fw_freq)

def get_max_amp_from_fft_windows(signal, window, sampFreq):

    file_max_amp = 0

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

        # Corta iniciando array (Tira ruído)
        fft_spectrum_abs = fft_spectrum_abs[2:len(fft_spectrum_abs)]

        # Filter
        n = 5  # the larger n is, the smoother curve will be
        b = [1.0 / n] * n
        a = 1
        filtered_spectrum_abs = filtfilt(b, a, fft_spectrum_abs)
        # filtered_spectrum_abs = butter_lowpass_filter(fft_spectrum_abs, cutoff, fs, order)

        window_max = max(filtered_spectrum_abs)

        if window_max > file_max_amp:
            file_max_amp = window_max

    return file_max_amp