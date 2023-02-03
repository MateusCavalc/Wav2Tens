
# FILTRO FIR - FREQ DE AMOSTRAGEM SIMULADA
# TESTAR COM E SEM OVERLAP

import sys

# Mono/Many algorithms
import mono
import many
import test

from utils import *

def free_mode():
    repeat = True

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('192.168.4.1', 5000)
    print('\n[Connecting to {0}] ...'.format(server_address))
    sock.connect(server_address)
    print('[Connected]\n')
    sock.sendall(b"command")
    time.sleep(1000 / 1000)

    uptime = 1000
    # downtime = 500
    #         

    for i in range(81):
        command = "1.0.1;1.0.1;1.{}.10".format(i)
        sock.sendall(command.encode('utf-8'))


    time.sleep(100 / 1000)
    sock.sendall(b"quit")
    time.sleep(1000 / 1000)
    sock.close()

def print_help():
    print(" Tradução de arquivo .wav para acionamento transcutâneo.\n")
    print(" uso:  wav2tens [-mono|-many] [arquivos .wav]\n")
    print("   modos:")
    print("           mono: Processamento de arquivo .wav e separação em regiões (baixas, médias e altas frequências).")
    print("           many: Processamento de até 3 arquivos .wav e ativação em canais separados.")
    print()

def print_invalid_mode(mode):
    print(" [!] MOVO INVÁLIDO - '{0}'".format(mode))
    print("")
    print("> Tente '-mono', '-mono-real' ou '-many'.\n")

print(" ______________________")
print(" |                      |")
print(" |  WAV2TENS CONVERTER  |")
print(" |______________________|\n")

args = sys.argv[1:]

if len(args) == 0:
    print_help()
    sys.exit()

mode = args[0]

# Remove flag modo
args.pop(0)

if len(args) > 0 and args[len(args) - 1] == '-d':
    debug = True
    # Remove flag debug
    args.pop()
else:
    debug = False

# Mono mode
if mode == '-mono':
    if len(args) == 0:
        print("\n [!] ARGUMENTOS INVÁLIDOS - Forneça um arquivo .wav para processamento.\n")
        sys.exit()

    filepath = args[0]
    print("# MODE: MONO")
    print("# File: {}".format(args))
    print("# Debug: {}".format(debug))
    mono.mono_mode(filepath, debug)

# Mono RealTime mode
elif mode == '-mono-real':
    print("# MODE: MONO REALTIME")
    print("# Debug: {}".format(debug))
    mono.mono_mode_realtime(debug)

# Many mode
elif mode == "-many":
    if len(args) == 0:
        print("\n [!] ARGUMENTOS INVÁLIDOS - Forneça pelo menos 1 arquivo .wav (máx 3) para processamento em canais separados.\n")
        sys.exit()

    print("# MODE: MANY")
    print("# Files: {}".format(args))
    print("# Debug: {}".format(debug))
    many.many_mode(args, debug)

# Test mode
elif mode == "-test":
    print("# MODE: TEST")
    test.test_mode()

# Modo livre - Para comandos soltos
elif mode == "-free":
    free_mode()

else:
    print_invalid_mode(mode)
    sys.exit()
