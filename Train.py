
import random
import datetime

from Scacc import *
from Games import *
from Eval import *
from NeuralNet import *

import numpy as np
import os.path
import json

import matplotlib.pyplot as plt
import threading
import pickle
import bz2


class TRAIN:

    def __init__(self, use_pytorch, Size, net_file=None, state_file=None, esempi_file=None, LearnRate=0.001, BatchSize=256, Epochs=10, NumGames=200, Depth=4, DaVuote=14, NetGen=1, AllGames=0, Filters=144, NetDepth=12, Verbose=1, Komi=0, numMCTSSims=30, LogPath=None, NThreads=1, EsempiMax=200000, MemoryPath=None, CPUct=1.0, SceltaTemp=1.0, ProbsTemp=2.0, Samples=10, GPUBatch=8):
        
        self.use_pytorch = use_pytorch

        # definisco il tipo di scacchiera su cui fare le partite
        self.Size = Size

        self.StatePath = state_file
        self.MemoryPath = MemoryPath
        self.EsempiPath = esempi_file
        
        # parametri per l'apprendimento
        self.learn_rate = LearnRate
        self.l2a = 0
        self.l2k = 0
        self.batch_size = BatchSize  # mini-batch size per l'apprendimento
        self.epochs = Epochs         # numero di passi per ogni apprendimento
        self.game_batch_number = NumGames
        self.EsempiMax = EsempiMax
        self.game_depth = Depth    # La profondità di valutazione usata durante la generazione delle partite
        self.DaVuote = DaVuote
        self.game_gen = NetGen     # Generazione dell NN con cui è stato valutato quel valore, 0 sono finali
                                   # perfetti
        self.Komi = Komi
        self.numMCTSSims = numMCTSSims
        self.CPUct = CPUct
        self.SceltaTemp = SceltaTemp
        self.ProbsTemp = ProbsTemp
        self.AllGames = AllGames
        self.Filters = Filters
        self.NetDepth = NetDepth
        self.LogPath = LogPath
        self.Samples = Samples
        self.GPUBatch = GPUBatch

        self.Verbose = Verbose

        if use_pytorch == 1:
            self.NeuralNet = NEURALNETPytorch(Size, net_file, self.learn_rate, self.l2a, self.l2k, self.Filters, self.NetDepth)
        else:
            self.NeuralNet = NEURALNET(Size, net_file, self.learn_rate, self.l2a, self.l2k, self.Filters, self.NetDepth)

        self.board_batch = list()
        self.Mosse_batch = list()
        self.pedine_batch = list()
        self.komi_batch = list()
        self.winlose_batch = list()

        self.lock = threading.Lock()
        self.NThreads = NThreads




    def TrainNet(self, Modo=0, AzzeraCicli=0):


        NCicli = 0
        TempoP = datetime.datetime.now()

        self.Games = GAMES(self.Size, self.NeuralNet, self.DaVuote)
        Scacc = SCACC(self.Size)
        Eval = EVAL(self.Size, self.NeuralNet)

        HistoryLoss = list()

        if AzzeraCicli == 0:
            if self.StatePath != None and os.path.exists(self.StatePath):
                with open(self.StatePath) as json_file:
                    data = json.load(json_file)
                    for p in data['Stati']:
                        NCicli = p['NCicli']
                        HistoryLoss = p['HistoryLoss']


        if len(HistoryLoss) > 4 and self.Verbose == 2:
            x = [x for x in range(3, len(HistoryLoss))]
            xp = [xp for xp in range(len(HistoryLoss) // 4 * 3, len(HistoryLoss))]
            plt.plot(x, HistoryLoss[3:])
            z = np.polyfit(xp, HistoryLoss[len(HistoryLoss) // 4 * 3:], 1)
            p = np.poly1d(z)
            plt.plot(xp,p(xp),"r--")
            plt.show()

        print()

        # Ricarico tutti gli esempi che avevo in memoria
        if (NCicli > 0):
            if self.MemoryPath != None and os.path.exists(self.MemoryPath):
                with bz2.BZ2File(self.MemoryPath, "rb") as fp:   # Unpickling
                    self.board_batch = pickle.load(fp)
                    self.Mosse_batch = pickle.load(fp)
                    self.pedine_batch = pickle.load(fp)
                    self.komi_batch = pickle.load(fp)
                    self.winlose_batch = pickle.load(fp)
            


        while True:

            NCicli += 1
            print("")
            print("Inizio Ciclo", NCicli)

            TempoA = datetime.datetime.now()
            TempoT = TempoA - TempoP            
            print("Tempo Trascorso: %02dh:%02dm:%02ds" % (TempoT.total_seconds() // 3600, (TempoT.total_seconds() // 60) % 60, TempoT.total_seconds() % 60))

            # Genero le partite e creo una lista
            # dello stato della board, pedine chi muove, pedine avversario,
            # mosse possibili


           


            Threads = list()
            for i in range(self.NThreads-1):
                Threads.append( threading.Thread(target=self.ThreadGeneraPartite, args=(i+1,)) )
                Threads[i].start()




            board_batch, Mosse_batch, pedine_batch, komi_batch, winlose_batch, NPartite = self.Games.GeneraPartiteMCTS(self.game_batch_number, self.game_depth, self.EsempiPath, self.game_gen, self.Komi, self.numMCTSSims, CPUct=self.CPUct, SceltaTemp=self.SceltaTemp, ProbsTemp=self.ProbsTemp, LogPath=self.LogPath, NumThread=0, GPUBatch=self.GPUBatch)

            NumEsempiGenerati = len(board_batch)

            if NPartite < self.game_batch_number and Modo != 0:
                Modo = 0
                NCicli -= 1
                continue

            # Inserisco tutti gli esempi creati nella lista totale
            with self.lock:
                self.board_batch.extend(board_batch)
                self.Mosse_batch.extend(Mosse_batch)
                self.pedine_batch.extend(pedine_batch)
                self.komi_batch.extend(komi_batch)
                self.winlose_batch.extend(winlose_batch)


            # Attendo che tutti i Threads finiscano
            for i in range(len(Threads)):
                Threads[i].join()
            


            # Elimino tutti gli Esempi più vecchi fino ad avere sempre un massimo di EsempiMax
            if len(self.board_batch) > self.EsempiMax:
                self.board_batch = self.board_batch[(len(self.board_batch) - self.EsempiMax):]
                self.Mosse_batch = self.Mosse_batch[(len(self.Mosse_batch) - self.EsempiMax):]
                self.pedine_batch = self.pedine_batch[(len(self.pedine_batch) - self.EsempiMax):]
                self.komi_batch = self.komi_batch[(len(self.komi_batch) - self.EsempiMax):]
                self.winlose_batch = self.winlose_batch[(len(self.winlose_batch) - self.EsempiMax):]




            self.VisualizzaVuoteEsempi(self.board_batch)

            # Metto l'input nel formato Esempi, Sizex, Sizey, colori
            board_batch = np.reshape(self.board_batch, [-1, self.Size, self.Size, 1])
            Mosse_batch = np.reshape(self.Mosse_batch, [-1, self.Size * self.Size])
            pedine_batch = np.reshape(self.pedine_batch, [-1, 1])
            komi_batch = np.reshape(self.komi_batch, [-1, 1])
            winlose_batch = np.reshape(self.winlose_batch, [-1, 1])

            
            #Prendo Samples posizioni a caso dagli esempi appena inseriti
            list_sample = random.sample(range(len(board_batch) - NumEsempiGenerati, len(board_batch)), self.Samples)

            #Visualizzo prima di apprenderle le attuale predizioni su tali esempi
            for i in list_sample:

                #k = i
                #Trovato = False
                #while not Trovato and k < len(self.board_batch):
                #    Vuote = self.ContaVuoteBoard(self.board_batch[k])
                #    if Vuote == 9 and abs(self.pedine_batch[k][0]) <= 6:
                #        Trovato = True
                #        i = k
                #    else:
                #        k += 1

                s_board_batch = board_batch[i]   
                s_komi_batch = komi_batch[i]

                p_Mosse_batch, p_winlose_batch, p_alpha_batch, p_beta_batch = self.NeuralNet.predici_board(s_board_batch, s_komi_batch)
                self.VisualizzaBoardMosse(np.reshape(board_batch[i], [self.Size, self.Size]), np.reshape(Mosse_batch[i], [self.Size, self.Size]), pedine_batch[i][0], np.reshape(p_Mosse_batch, (self.Size, self.Size)), komi_batch[i][0], winlose_batch[i][0], p_alpha_batch[0][0], p_beta_batch[0][0], p_winlose_batch[0][0])

                Scacc.ImpostaScaccDaBoard(np.reshape(board_batch[i], [self.Size, self.Size]))
                Eval.Valuta_Scacc(Scacc, self.game_depth, 1)


            if self.use_pytorch == 1:
                EvalData = self.NeuralNet.evaluate(board_batch, Mosse_batch, pedine_batch, komi_batch, batch_size=self.batch_size)
            else:
                EvalData = self.NeuralNet.model.evaluate([board_batch, komi_batch], [Mosse_batch, winlose_batch, pedine_batch, winlose_batch], batch_size=self.batch_size, verbose = 2)

            print("Init Loss:", EvalData[0])

            HistoryLoss.append(EvalData[0])

            if len(HistoryLoss) > 4 and self.Verbose == 2:
                x = [x for x in range(3, len(HistoryLoss))]
                xp = [xp for xp in range(len(HistoryLoss) // 4 * 3, len(HistoryLoss))]
                plt.plot(x, HistoryLoss[3:])
                z = np.polyfit(xp, HistoryLoss[len(HistoryLoss) // 4 * 3:], 1)
                p = np.poly1d(z)
                plt.plot(xp,p(xp),"r--")
                plt.show()


            if self.use_pytorch == 1:
                History = self.NeuralNet.train(board_batch, Mosse_batch, pedine_batch, komi_batch, batch_size=self.batch_size, epochs=self.epochs)
            else:
                History = self.NeuralNet.model.fit([board_batch, komi_batch], [Mosse_batch, winlose_batch, pedine_batch, winlose_batch], batch_size=self.batch_size, epochs=self.epochs, verbose = 2, shuffle = True)


            # Salvo tutti i nuovi valori della Rete           
            self.NeuralNet.Salva()
            self.NeuralNet.Salva(self.NeuralNet.net_file.replace(".keras", ("_Gen%d.keras" % NCicli)))


            # Salvo tutte le partite che ho in memoria
            with bz2.BZ2File(self.MemoryPath, "wb") as fp:   #Pickling
                pickle.dump(self.board_batch, fp)
                pickle.dump(self.Mosse_batch, fp)
                pickle.dump(self.pedine_batch, fp)
                pickle.dump(self.komi_batch, fp)
                pickle.dump(self.winlose_batch, fp)


            data = {}
            data['Stati'] = []
            data['Stati'].append({ 'NCicli': NCicli,
                                   'HistoryLoss': HistoryLoss })
            with open(self.StatePath, 'w') as outfile:
                json.dump(data, outfile)



            #Visualizzo dopo averle apprese le attuali predizioni sugli stessi 10 esempi presi a caso
            for i in list_sample:                  

                s_board_batch = board_batch[i]   
                s_komi_batch = komi_batch[i]

                p_Mosse_batch, p_winlose_batch, p_alpha_batch, p_beta_batch = self.NeuralNet.predici_board(s_board_batch, s_komi_batch)
                self.VisualizzaBoardMosse(np.reshape(board_batch[i], [self.Size, self.Size]), np.reshape(Mosse_batch[i], [self.Size, self.Size]), pedine_batch[i][0], np.reshape(p_Mosse_batch, (self.Size, self.Size)), komi_batch[i][0], winlose_batch[i][0], p_alpha_batch[0][0], p_beta_batch[0][0], p_winlose_batch[0][0])

                Scacc.ImpostaScaccDaBoard(np.reshape(board_batch[i], [self.Size, self.Size]))
                Eval.Valuta_Scacc(Scacc, self.game_depth, 1)
 



        TempoA = datetime.datetime.now()
        TempoT = TempoA - TempoP            

        print("Finito in: %02dh:%02dm:%02ds" % (TempoT.total_seconds() // 3600, (TempoT.total_seconds() // 60) % 60, TempoT.total_seconds() % 60))
        
        return









    def ContaVuoteBoard(self, board):

        Vuote = 0

        for y in range(0, self.Size):
            for x in range(0, self.Size):
                if (board[y][x] == 0):
                    Vuote += 1

        return Vuote



    def VisualizzaVuoteEsempi(self, boards):

        print("Vuote Esempi")

        VuoteEsempi = [0] * (self.Size*self.Size + 1)
        Maxv = 0
        Minv = 200

        for i in range(len(boards)):
            v = self.ContaVuoteBoard(boards[i])
            VuoteEsempi[v] += 1

            if v > Maxv:
                Maxv = v
            if v < Minv:
                Minv = v

        for i in range(Minv, Maxv+1):
            print("%5d: " % i + str(VuoteEsempi[i]))



    def VisualizzaBoardMosse(self, board, mosse, ris_reale, mosse_pred, komi, winlose_reale, ris_pred, Beta_pred, winlose_pred):

        NPezzi = 0
        NPezziCm = 0
        NPezziAv = 0

        for y in range(0, self.Size):
            for x in range(0, self.Size):
                if (board[y][x] != 0):
                    NPezzi += 1
                    if (board[y][x] == 1):
                        NPezziCm += 1
                    else:
                        NPezziAv += 1

        if NPezzi % 2 == 0:
            ChiMuove = 1
            PedinaCm = " X "
            PedinaAv = " O "
        else:
            ChiMuove = 2
            PedinaCm = " O "
            PedinaAv = " X "

        Pezzi = [0, 0, 0]
        Pezzi[ChiMuove] = NPezziCm
        Pezzi[3 - ChiMuove] = NPezziAv
    
        print("")
        Riga = " "
        RigaM = " "
        RigaP = " "
        for x in range(self.Size):
            Riga += " " + chr(65 + x) + " "
            RigaM += "    " + chr(65 + x) + ""
            RigaP += "    " + chr(65 + x) + ""
        print(Riga + "" + RigaM + "      " + RigaP)
        
        for y in range(0, self.Size):
            Riga = str(y + 1)
            RigaP = ""
            RigaM = ""
            for x in range(0, self.Size):
                if mosse_pred[y][x] >= 0.01:
                    RigaP += " %1.2f" % mosse_pred[y][x]
                else:
                    RigaP += "   . "

                if mosse[y][x] >= 0.01:
                    RigaM += " %1.2f" % mosse[y][x]
                else:
                    RigaM += "   . "

                if (board[y][x] == 1):
                    Riga += PedinaCm
                elif  (board[y][x] == -1):
                    Riga += PedinaAv
                else: 
                    if (mosse[y][x] == 1):
                        Riga += " ! "
                    else:
                        Riga += " . "
            print(Riga + "  " + RigaM + "       " + RigaP)
            
        print("V:", (self.Size * self.Size) - NPezzi, " Pn:", Pezzi[1], "Pb:", Pezzi[2])     

        print("Ris.reale: %+6.2f" % ris_reale, " " * (self.Size-6)*3, "    Komi: %+1.2f" % komi, "  WLD.reale: %+1.2f" % winlose_reale, " " * (self.Size-6)*3, "       A: %+1.2f" % ris_pred, "B: %+1.2f" % Beta_pred, " WDL.pred: %+1.2f" % winlose_pred)





    def ThreadGeneraPartite(self, NumThread):

         board_batch, Mosse_batch, pedine_batch, komi_batch, winlose_batch, NPartite = self.Games.GeneraPartiteMCTS(self.game_batch_number, self.game_depth, self.EsempiPath, self.game_gen, self.Komi, self.numMCTSSims, CPUct=self.CPUct, SceltaTemp=self.SceltaTemp, ProbsTemp=self.ProbsTemp, LogPath=self.LogPath, NumThread=NumThread, GPUBatch=self.GPUBatch)

         # Inserisco tutti gli esempi creati nella lista totale
         with self.lock:
             self.board_batch.extend(board_batch)
             self.Mosse_batch.extend(Mosse_batch)
             self.pedine_batch.extend(pedine_batch)
             self.komi_batch.extend(komi_batch)
             self.winlose_batch.extend(winlose_batch)


           




