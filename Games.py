
import random
import datetime

from Board import *
from Scacc import *
from Eval import *
from UCTS import *

import numpy as np
from tqdm import tqdm
import ast
import math


class GAMES:

    def __init__(self, Size, NeuralNet, PartitaDaVuote = 31):
        
        # definisco il tipo di scacchiera su cui fare le partite
        self.Size = Size
        self.NeuralNet = NeuralNet

        self.File = None

        self.PartitaDaVuote = PartitaDaVuote

        self.EsempiLetti = 0




    def GeneraPartiteMCTS(self, NPartiteMax, Prof, EsempiPath=None, NumGen = 1, KomiMax = 0, numMCTSSims = 30, CPUct = 1.0, SceltaTemp = 1.0, ProbsTemp = 2.0, LogPath=None, NumThread=0, GPUBatch=8):

        Eval = EVAL(self.Size, self.NeuralNet)
        Scacc = SCACC(self.Size)

        # Ogni elemento della lista contiene i 3 stati della board
        boards = list()
        mosse = list()
        pedine = list()
        komis = list()
        winlose = list()


        print("")
        print("Thread", NumThread, " Generando", NPartiteMax, "Partite...")

    
        TempoP = datetime.datetime.now()
        TempoA = datetime.datetime.now()
        TempoT = TempoA - TempoP

        SceltaTempORG = SceltaTemp
        
        Ok = True
        NPartite = 0
        for NPartite in tqdm(range(0, NPartiteMax), disable=(NumThread > 0)):     
            Scacc.NMossa = 1
            NMossaMin = 1

            SceltaTemp = SceltaTempORG

            # Il Komi scelto random per il NERO
            Komi = random.randint(-KomiMax, +KomiMax)
            Komi = int(Komi/2) * 2

            ts = UCTS(self.NeuralNet, None, numMCTSSims, GPUBatch)

            # Gioco fino a 9 mosse dalla fine o fino a che non è finita la partita
            while Ok and Scacc.Table[Scacc.NMossa].ChiMuove != 0 and Scacc.NMossa < Board.MossaFinalePerfetto:
                
                Vuote = Scacc.MossaFinale - Scacc.NMossa

                if Scacc.NMossa >= Scacc.MossaFinale - self.PartitaDaVuote and Scacc.NMossa < Board.MossaFinalePerfetto:

                    # Dichiaro i 2 piani bidimensionali contenenti nelle caselle,
                    # pedine di ChiMuove, pedine Avversario
                    MtChiMuove = np.zeros(shape=(self.Size, self.Size), dtype=int)
                    
                    # Riempio il piano con le pedine
                    for y in range(1, self.Size + 1):
                        for x in range(1, self.Size + 1):
                            if (Scacc.Table[Scacc.NMossa].B[y * (self.Size+2) + x] == Scacc.Table[Scacc.NMossa].ChiMuove):
                                MtChiMuove[y - 1][x - 1] = 1
                            elif (Scacc.Table[Scacc.NMossa].B[y * (self.Size+2) + x] == 3 - Scacc.Table[Scacc.NMossa].ChiMuove):
                                MtChiMuove[y - 1][x - 1] = -1

                    

                    tSceltaTemp = SceltaTemp
                    tCPUct = CPUct
                    tKomi = Komi

                    #if (Vuote <= 20):
                    #    tSceltaTemp /= 1.2

                    #if (Vuote <= 15):
                    #    tSceltaTemp /= 1.5

                    if (Vuote <= 10):
                        tSceltaTemp = 0
                        tCPUct = 0

                    Mossa, Best, Val, Alpha, MtMosse = ts.suggest_move(None, tCPUct, tSceltaTemp, ProbsTemp, Komi)

                    if Scacc.Table[Scacc.NMossa].ChiMuove == 2:
                        tKomi = -Komi


                    # Se non gioco la mossa migliore allora scarto tutti gli esempi da questo ai precedenti e poi gioco solo la mossa migliore                
                    if (Mossa != Best):
                        NMossaMin = Scacc.NMossa + 1 
                        SceltaTemp /= 2.0


                    
                    MtMosse = np.reshape(MtMosse, (self.Size, self.Size))
                       
                    boards.append(MtChiMuove)
                    mosse.append(MtMosse)
                    pedine.append(Alpha)
                    komis.append(tKomi)
                    winlose.append(Val)
                    for r in range(1, 4):
                        boards.append(np.rot90(MtChiMuove, r))
                        mosse.append(np.rot90(MtMosse, r))
                        pedine.append(Alpha)
                        komis.append(tKomi)
                        winlose.append(Val)
                        

                    # Ribalto la scacchiera da sinistra a destra
                    rMtChiMuove = np.zeros(shape=(self.Size, self.Size), dtype=int)
                    rMtMosse = np.zeros(shape=(self.Size, self.Size), dtype=float)
                    for y in range(0, self.Size):
                        for x in range(0, self.Size):
                            rMtChiMuove[y][x] = MtChiMuove[(self.Size - 1) - y][x]
                            rMtMosse[y][x] = MtMosse[(self.Size - 1) - y][x]

                    boards.append(rMtChiMuove)
                    mosse.append(rMtMosse)
                    pedine.append(Alpha)
                    komis.append(tKomi)
                    winlose.append(Val)
                    for r in range(1, 4):
                        boards.append(np.rot90(rMtChiMuove, r))
                        mosse.append(np.rot90(rMtMosse, r))
                        pedine.append(Alpha)
                        komis.append(tKomi)
                        winlose.append(Val)
                        
                            
                else:
                    MosseL = Scacc.Table[Scacc.NMossa].DammiMosse()

                    iScelta = random.randint(0, len(MosseL)-1)
                    Mossa = MosseL[iScelta]

                    NMossaMin = Scacc.NMossa + 1


                
                Ok = Scacc.FaiMossa(Mossa)
                ts.PlayRootMove(Mossa)

                if not Ok:
                    print("Non si puo fare la mossa")




            if Scacc.Table[Scacc.NMossa].ChiMuove == 0:
                Val = Scacc.Table[Scacc.NMossa].NPezzi[1] - Scacc.Table[Scacc.NMossa].NPezzi[2]
            else:
                Mossa, Val = Eval.Valuta_Scacc(Scacc, 1)
                if Scacc.Table[Scacc.NMossa].ChiMuove == 2:
                    Val = -Val

            # Per tutte le posizioni da cui è stata eseguita sempre la mossa migliore backpropago il risultato finale reale
            Offs = 0
            Last = len(pedine) - 1
            for i in range(Scacc.NMossa-1, NMossaMin-1, -1):

                tVal = Val
                tKomi = Komi
                if Scacc.Table[i].ChiMuove == 2:
                    tVal = -Val
                    tKomi = -Komi
                
                winl = math.tanh((tVal-tKomi) / 4)

                for k in range(8):
                    pedine[Last - Offs] = tVal
                    winlose[Last - Offs] = winl
                    Offs += 1
                    

            Sequenza = ""
            for i in range(1, Scacc.NMossa):
                Sequenza += chr(65 + Scacc.Seq[i]%(self.Size+2) - 1) + str(int(Scacc.Seq[i]/(self.Size+2)))
            Sequenza += " %+3d" % Val
            Sequenza += "  Komi:%+3d" % Komi
            Sequenza += "  MMin:%d" % NMossaMin
            Sequenza += "\n"


            if EsempiPath != None:
                with open(EsempiPath, 'a') as file:
                    file.write(Sequenza)

             



        TempoA = datetime.datetime.now()
        TempoT = TempoA - TempoP            

        print()
        print("Thread", NumThread, "Partite:", NPartite, " Sec: %1.2f" % TempoT.total_seconds(), " P/s: %1.2f" % (NPartite / (TempoT.total_seconds() + 0.0001)))
        

        return boards, mosse, pedine, komis, winlose, NPartite





    def VisualizzaBoardMosse(self, Size, board, mosse, ris_reale, mosse_pred, komi, winlose_reale, ris_pred, Beta_pred, winlose_pred):

        NPezzi = 0
        NPezziCm = 0
        NPezziAv = 0

        for y in range(0, Size):
            for x in range(0, Size):
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
        for x in range(Size):
            Riga += " " + chr(65 + x) + " "
            RigaM += "    " + chr(65 + x) + ""
            RigaP += "    " + chr(65 + x) + ""
        print(Riga + "" + RigaM + "      " + RigaP)
        
        for y in range(0, Size):
            Riga = str(y + 1)
            RigaP = ""
            RigaM = ""
            for x in range(0, Size):
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
            
        print("V:", (Size * Size) - NPezzi, " Pn:", Pezzi[1], "Pb:", Pezzi[2])     

        print("Ris.reale: %+6.2f" % ris_reale, "    Komi: %+1.2f" % komi, "  WLD.reale: %+1.2f" % winlose_reale, "       A: %+1.2f" % ris_pred, "B: %+1.2f" % Beta_pred, " WDL.pred: %+1.2f" % winlose_pred)
