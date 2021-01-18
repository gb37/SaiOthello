
from Board import *
from NeuralNet import *

import numpy as np



class SCACC:
    
    def __init__(self, Size):
        
        self.Size = Size
        self.Table = []
        self.Seq = [0] * Size*Size

        MosseMax = Size * Size
        for i in range(MosseMax):
            self.Table.append(BOARD())
            self.Table[i].NMossa = i
            self.Table[i].Inizializza()

        self.MossaFinale = MosseMax - 3
        
        self.NMossa = 1
        

    def stringRepresentation(self, Komi=0):
        s = str(Komi) + str(self.Table[self.NMossa].ChiMuove)
        for y in range(1, self.Size+1):
            for x in range(1, self.Size+1):
                s += str(self.Table[self.NMossa].B[y * (self.Size+2) + x])
        return s


    def Visualizza(self):
    
        print("")

        Riga = " "
        for x in range(self.Size):
            Riga += " " + chr(65 + x) 
        print(Riga)
        
        for y in range(1, self.Size + 1):
            Riga = str(y)
            for x in range(1, self.Size + 1):
                if (self.Table[self.NMossa].B[y * (self.Size+2) + x] == 1):
                    Riga += " X"
                elif  (self.Table[self.NMossa].B[y * (self.Size+2) + x] == 2):
                    Riga += " O"
                else: 
                    Riga += " -"
            print(Riga)
            
        print("V:", self.MossaFinale - self.NMossa, " Pn:", self.Table[self.NMossa].NPezzi[1], "Pb:", self.Table[self.NMossa].NPezzi[2])     
        
    

    def FaiMossa(self, Pos, NonTestare=False):
        
        if NonTestare or self.Table[self.NMossa].ChiMuove != 0 and self.Table[self.NMossa].B[Pos] == 0 and self.Table[self.NMossa].ProvaMossa(Pos):
            for i in range(self.Size+2, (self.Size+2)*(self.Size+2) - (self.Size+2)):
                self.Table[self.NMossa + 1].B[i] = self.Table[self.NMossa].B[i]

            self.Table[self.NMossa + 1].ChiMuove = self.Table[self.NMossa].ChiMuove
            self.Table[self.NMossa + 1].NPezzi[1] = self.Table[self.NMossa].NPezzi[1]
            self.Table[self.NMossa + 1].NPezzi[2] = self.Table[self.NMossa].NPezzi[2]
                
            self.Seq[self.NMossa] = Pos

            self.NMossa += 1
            self.Table[self.NMossa].EseguiMossa(Pos)

            # Controllo che ci sia una mossa disponibile altrimenti o passo o
            # dichiaro finita la partita

            if not self.Table[self.NMossa].TrovaMossa(self.Table[self.NMossa].ChiMuove):
                if not self.Table[self.NMossa].TrovaMossa(3 - self.Table[self.NMossa].ChiMuove):
                    self.Table[self.NMossa].ChiMuove = 0
                else:
                    self.Table[self.NMossa].ChiMuove = 3 - self.Table[self.NMossa].ChiMuove

            return True
            
        return False
        
        
        
    def DammiBoardDaPredire(self):

        # Dichiaro il piano bidimensionale contenente nelle caselle
        # pedine di ChiMuove, pedine Avversario
        MtChiMuove = np.zeros(shape=(self.Size, self.Size))

        # Riempio i primi 2 piani con le pedine
        for y in range(1, self.Size + 1):
            for x in range(1, self.Size + 1):
                if (self.Table[self.NMossa].B[y * (self.Size+2) + x] == self.Table[self.NMossa].ChiMuove):
                    MtChiMuove[y - 1][x - 1] = 1.0
                elif (self.Table[self.NMossa].B[y * (self.Size+2) + x] == 3 - self.Table[self.NMossa].ChiMuove):
                    MtChiMuove[y - 1][x - 1] = -1.0

        return MtChiMuove



    def ImpostaScaccDaBoard(self, board):

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
        else:
            ChiMuove = 2


        Pezzi = [0, 0, 0]
        Pezzi[ChiMuove] = NPezziCm
        Pezzi[3 - ChiMuove] = NPezziAv

        NMossa = NPezzi - 3

        for y in range(0, self.Size):
            for x in range(0, self.Size):
                if (board[y][x] == 1):
                    self.Table[NMossa].B[(y+1) * (self.Size+2) + x+1] = ChiMuove
                elif (board[y][x] == -1):
                    self.Table[NMossa].B[(y+1) * (self.Size+2) + x+1] = 3 - ChiMuove
                else:
                    self.Table[NMossa].B[(y+1) * (self.Size+2) + x+1] = 0
        
        self.NMossa = NMossa
        self.Table[NMossa].ChiMuove = ChiMuove
        self.Table[NMossa].NPezzi[1] = Pezzi[1]
        self.Table[NMossa].NPezzi[2] = Pezzi[2]



    def ImpostaSequenza(self, Seq):

        Seq = Seq.upper()

        self.NMossa = 1

        Ok = True
        k = 0
        while Ok and k < len(Seq):
            x = ord(Seq[k]) - ord('A') + 1
            y = ord(Seq[k+1]) - ord('1') + 1

            Ok = self.FaiMossa(y*(self.Size+2) + x)

            k += 2




    def VisualizzaBoardMosse(self, mosse_pred, probs, komi, ris_pred, Beta_pred, winlose_pred):

        board = self.DammiBoardDaPredire()

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

                if probs[y][x] >= 0.01:
                    RigaM += " %1.2f" % probs[y][x]
                else:
                    RigaM += "   . "

                if (board[y][x] == 1):
                    Riga += PedinaCm
                elif  (board[y][x] == -1):
                    Riga += PedinaAv
                else: 
                    Riga += " . "
            print(Riga + "  " + RigaM + "       " + RigaP)
            
        print("V:", (self.Size * self.Size) - NPezzi, " Pn:", Pezzi[1], "Pb:", Pezzi[2])     

        print("Komi: %+1.2f" % komi, "                                              A: %+1.2f" % ris_pred, "B: %+1.2f" % Beta_pred, " WDL.pred: %+1.2f" % winlose_pred)
