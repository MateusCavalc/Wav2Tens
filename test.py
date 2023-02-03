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
NEW_FREQ_SCALE_MIN = 1
NEW_FREQ_SCALE_MAX = 80
pace = 1

def mute_channels(sock):
    sock.sendall("1.0.1;1.0.1;1.0.1".encode('utf-8'))
    time.sleep(100 / 1000)

def test0(sock):
    repeat = True

    active_time = 1000 # millis
    fixed_duty = 5 # %
    fixed_freq = 50 # Hz

    while repeat:
        # Canal 1
        # Subindo o duty
        for test_duty in range(TENS_MAX_MOD + 1):
            command = "1.{:02d}.{:02d};1.00.01;1.00.01".format(test_duty, fixed_freq)
            print(command)
            sock.sendall(command.encode('utf-8'))
            time.sleep(active_time / 1000)

        # Descendo o duty
        for test_duty in range(TENS_MAX_MOD, -1, -1):
            command = "1.{:02d}.{:02d};1.00.01;1.00.01".format(test_duty, fixed_freq)
            print(command)
            sock.sendall(command.encode('utf-8'))
            time.sleep(active_time / 1000)

        # Canal 2
        # Subindo o duty
        for test_duty in range(TENS_MAX_MOD + 1):
            command = "1.00.01;1.{:02d}.{:02d};1.00.01".format(test_duty, fixed_freq)
            print(command)
            sock.sendall(command.encode('utf-8'))
            time.sleep(active_time / 1000)

        # Descendo o duty
        for test_duty in range(TENS_MAX_MOD, -1, -1):
            command = "1.00.01;1.{:02d}.{:02d};1.00.01".format(test_duty, fixed_freq)
            print(command)
            sock.sendall(command.encode('utf-8'))
            time.sleep(active_time / 1000)

        # Canal 3
        # Subindo o duty
        for test_duty in range(TENS_MAX_MOD + 1):
            command = "1.00.01;1.00.01;1.{:02d}.{:02d}".format(test_duty, fixed_freq)
            print(command)
            sock.sendall(command.encode('utf-8'))
            time.sleep(active_time / 1000)

        # Descendo o duty
        for test_duty in range(TENS_MAX_MOD, -1, -1):
            command = "1.00.01;1.00.01;1.{:02d}.{:02d}".format(test_duty, fixed_freq)
            print(command)
            sock.sendall(command.encode('utf-8'))
            time.sleep(active_time / 1000)

        mute_channels(sock)

        c = input("\n Repeat test? (y/n)\n\n >")

        if c == 'n':
            repeat = False

    repeat = True

    while repeat:
        # Canal 1
        # Subindo a frequência
        for test_freq in range(5, 81, 5):
            command = "1.{:02d}.{:02d};1.00.01;1.00.01".format(fixed_duty, test_freq)
            print(command)
            sock.sendall(command.encode('utf-8'))
            time.sleep(active_time / 1000)

        # Descendo a frequência
        for test_freq in range(80, 5, -5):
            command = "1.{:02d}.{:02d};1.00.01;1.00.01".format(fixed_duty, test_freq)
            print(command)
            sock.sendall(command.encode('utf-8'))
            time.sleep(active_time / 1000)

        # Canal 2
        # Subindo a frequência
        for test_freq in range(5, 81, 5):
            command = "1.00.01;1.{:02d}.{:02d};1.00.01".format(fixed_duty, test_freq)
            print(command)
            sock.sendall(command.encode('utf-8'))
            time.sleep(active_time / 1000)

        # Descendo a frequência
        for test_freq in range(80, 5, -5):
            command = "1.00.01;1.{:02d}.{:02d};1.00.01".format(fixed_duty, test_freq)
            print(command)
            sock.sendall(command.encode('utf-8'))
            time.sleep(active_time / 1000)

        # Canal 3
        # Subindo a frequência
        for test_freq in range(5, 81, 5):
            command = "1.00.01;1.00.01;1.{:02d}.{:02d}".format(fixed_duty, test_freq)
            print(command)
            sock.sendall(command.encode('utf-8'))
            time.sleep(active_time / 1000)

        # Descendo a frequência
        for test_freq in range(80, 5, -5):
            command = "1.00.01;1.00.01;1.{:02d}.{:02d}".format(fixed_duty, test_freq)
            print(command)
            sock.sendall(command.encode('utf-8'))
            time.sleep(active_time / 1000)

        mute_channels(sock)

        c = input("\n Repeat test? (y/n)\n\n > ")

        if c == 'n':
            repeat = False

def test1(sock):
    # Usuário não tenta acertar ordem
    filepaths = ['wav-samples/fundamentals-1-2-3-4.wav']
    test_many(sock, filepaths)
    filepaths = ['wav-samples/fundamentals-4-3-2-1.wav']
    test_many(sock, filepaths)

    # Usuário tenta acertar ordem
    filepaths = ['wav-samples/fundamentals-2-1-3-4.wav']
    test_many(sock, filepaths)
    filepaths = ['wav-samples/fundamentals-3-2-1-4.wav']
    test_many(sock, filepaths)

def test2(sock):
    global NEW_FREQ_SCALE_MIN
    global NEW_FREQ_SCALE_MAX

    NEW_FREQ_SCALE_MIN = 1
    NEW_FREQ_SCALE_MAX = 50
    filepaths = ['wav-samples/teste2_melodia.wav']
    test_many(sock, filepaths)

    # 3 Etapas
    # |- Ativação sem acompanhamento sonoro (Verifica se usuário percebe a melodia).
    # |- Ativação com acompanhamento sonoro.
    # |- Ativação sem acompanhamento sonoro (Verifica se usuário passa a perceber a melodia).
    filepaths = ['wav-samples/teste2_natal.wav']
    test_many(sock, filepaths)

def test3(sock):
    global NEW_FREQ_SCALE_MIN
    global NEW_FREQ_SCALE_MAX
    
    NEW_FREQ_SCALE_MIN = 30
    NEW_FREQ_SCALE_MAX = 80
    # Testa Ritmo Granular - Kick, hihat, snare, bass, ...
    # Feedback do usuásio sobre ritmo.
    filepaths = ['wav-samples/kick.wav']
    test_many(sock, filepaths)

    filepaths = ['wav-samples/silencio.wav', 'wav-samples/hihat.wav']
    test_many(sock, filepaths)

    filepaths = ['wav-samples/silencio.wav', 'wav-samples/silencio.wav', 'wav-samples/snare.wav']
    test_many(sock, filepaths)

    # Atualiza limite mínimo da nova escala tátil
    # Testa Ritmo Composto (Beat)
    filepaths = ['wav-samples/kick.wav', 'wav-samples/hihat.wav', 'wav-samples/snare.wav']
    test_many(sock, filepaths)

    # Testa Beat Melodico
    filepaths = ['wav-samples/kick.wav', 'wav-samples/kick-melodia.wav', 'wav-samples/hihat.wav']
    test_many(sock, filepaths)

    # Volta limite antigo
    NEW_FREQ_SCALE_MIN = 1

def test4(sock):
    # Teste mais complexo (O Dream)
    # Ritmo + Melodia
    # Verificar se usuário sente uma ativação ritmada.
    global NEW_FREQ_SCALE_MIN
    global NEW_FREQ_SCALE_MAX
    
    NEW_FREQ_SCALE_MIN = 1
    NEW_FREQ_SCALE_MAX = 50
    filepaths = ['wav-samples/batucada-tanta.wav']
    test_many(sock, filepaths)

    # Verificar melodia de solo de cavaco.
    filepaths = ['wav-samples/silencio.wav', 'wav-samples/melodia-cavaco.wav']
    test_many(sock, filepaths)

    # Testa ativação simultânea de múltiplos canais (tanta e cavaco).
    filepaths = ['wav-samples/batucada-tanta.wav', 'wav-samples/melodia-cavaco.wav']
    test_many(sock, filepaths)


def test_mode():
    # Abre conexão do socket com o ESP32 via WiFi
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('192.168.4.1', 5000)
    print('\n[Connecting to ESP32 - {0}] ...'.format(server_address))
    sock.connect(server_address)
    print('[Connected]\n')

    # Inicia modo de envio de comandos
    sock.sendall(b"command")
    time.sleep(1000 / 1000)

    # TESTE 0 - Variação de duty e frequência

    # test0(sock)

    # TESTE 1 - Vericação sensorial de frequências fundamentais puras
    #           (Permutação de ondas 100Hz, 250Hz, 440Hz e 1000Hz)

    # test1(sock)

    # TESTE 2 - Vericação sensorial de frequências fundamentais compostas
    #           (Melodias aplicadas em instrumentos)

    test2(sock)

    # TESTE 3 - Vericação sensorial de ritmos
    #           (Intrumentos de percussão e bases ritmadas compostas por kick, snare, hihat, etc..)

    test3(sock)

    # TESTE 4

    test4(sock)

    # Finaliza o socket
    time.sleep(100 / 1000)
    sock.sendall(b"quit")
    time.sleep(1000 / 1000)
    sock.close()

def test_many(sock, filepaths):
    global tens_amps
    global tens_freqs
    global filtered_tens_amps
    global filtered_tens_freqs
    global OLD_FREQ_SCALE_MINS
    global OLD_FREQ_SCALE_MAXS
    global subcommands
    global _delays
    global _audio_lens

    threads = list()

    for index, filepath in enumerate(filepaths):
        x = threading.Thread(target=process_channel, args=("canal_{0}".format(index+1), filepath, tens_amps[index], tens_freqs[index]))
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
    ax1.set_ylim([0, 1.1])
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
    ax1.set_ylim([0, TENS_MAX_MOD + 5])
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

        c = input("\n Repeat test? (y/n)\n\n > ")

        if c == 'n':
            repeat = False

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
    

def process_channel(threadName, filepath, tens_amp, tens_freq):
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

        mean_amp = mean(freq_peaks_amp)

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