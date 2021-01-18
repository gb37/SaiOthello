
import random

from Scacc import *
from NeuralNet import *

import numpy as np


class EVAL:

    def __init__(self, Size, NeuralNet):

        self.Size = Size

        self.Calcolate = 0
        self.NeuralNet = NeuralNet


    def CoordBoardToZero(self, Pos):

        y = int(Pos / (self.Size+2)) - 1
        x = Pos % (self.Size+2) - 1

        return y * self.Size + x



    def Valuta_Scacc(self, Scacc: SCACC, Prof, Modo=0):

        self.Calcolate = 0
        NMossa = Scacc.NMossa
        ValMax = -1000
        Alpha = -1000

        Vuote = Scacc.Table[NMossa].DammiVuote()

        # Ordino le mosse secondo le probabilità predette
        if Prof >= 2:
            board = Scacc.DammiBoardDaPredire()
            komi = np.array([float(0)])
            PMosse, winlose, ap, bp = self.NeuralNet.predici_board(board, komi)

            MosseL = Scacc.Table[NMossa].DammiMosseConVuote(Vuote)

            Mosse = list()
            for _ in range(len(MosseL)):
                ProbMax = -1000
                for i in range(len(MosseL)):
                    if (PMosse[0][ self.CoordBoardToZero(MosseL[i]) ] > ProbMax):
                        ProbMax = PMosse[0][ self.CoordBoardToZero(MosseL[i]) ]
                        iMax = i
                PMosse[0][ self.CoordBoardToZero(MosseL[iMax]) ] = -1000
                Mosse.append(MosseL[iMax])

        else:
            Mosse = Scacc.Table[NMossa].DammiMosseConVuote(Vuote)


        if Prof >= 2:
            # Predico tutte le mosse delle sottoscacchiere di ogni mossa
            # perche' predire piu' scacchiere contemporaneamente e' molto piu'
            # veloce rispetto ad una sola
            boards = list()
            komi = list()
            for i in range(len(Mosse)):
                # Ripristino NMossa che è stato incrementato da FaiMossa
                Scacc.NMossa = NMossa
                Scacc.FaiMossa(Mosse[i], True)
                board = Scacc.DammiBoardDaPredire()
                boards.append(board)
                komi.append([float(0)])

            mossep, valp, ap, bp = self.NeuralNet.predici_board(boards, komi)
        else:
            mossep = [0] * 40
            valp = [0] * 40

        ValMosse = ""
        for i in range(len(Mosse)):

            if (Modo == 1):
                Alpha = -1000

            # Ripristino NMossa che è stato incrementato da FaiMossa
            Scacc.NMossa = NMossa
            Scacc.FaiMossa(Mosse[i], True)
            self.Calcolate += 1

            Vp = Vuote.index(Mosse[i])
            Vuote.remove(Mosse[i])

            # Se il giocatore successivo sono sempre io allora devo prendere il
            # risultato dal mio punto di vista
            if Scacc.Table[NMossa].ChiMuove != Scacc.Table[NMossa + 1].ChiMuove:
                Val = self.Valuta_Mossa(Scacc, Vuote, NMossa + 1, Prof - 1, -(self.Size * self.Size), -Alpha, mossep[i], valp[i])
            else:
                Val = - self.Valuta_Mossa(Scacc, Vuote, NMossa + 1, Prof - 1, Alpha, +(self.Size * self.Size), mossep[i], valp[i])

            Vuote.insert(Vp, Mosse[i])

            if (Modo == 1):
                ValMosse += (chr(65 + Mosse[i]%(self.Size+2) - 1) + str(int(Mosse[i]/(self.Size+2))) + ":%+1.2f " % Val)


            if Val > ValMax:
                Alpha = Val
                ValMax = Val
                Scelta = Mosse[i]

        Scacc.NMossa = NMossa

        if (Modo == 1):
            print(ValMosse)

        return Scelta, ValMax


    
    def Valuta_Mossa(self, Scacc: SCACC, Vuote, NMossa, Prof, Alpha, Beta, MossePr=None, valpr=None):

        # Se sto vicino alla fine della partita allora calcolo direttamente il
        # FinalePerfetto
        if NMossa >= Scacc.MossaFinale - 8:
            return self.Valuta_Finale(Scacc, Vuote, NMossa, Alpha, Beta)

        # Se è finita la partita ritorno la differenza pedine dal punto di
        # vista del giocatore precedente
        if Scacc.Table[NMossa].ChiMuove == 0:
            return Scacc.Table[NMossa].NPezzi[Scacc.Table[NMossa - 1].ChiMuove] - Scacc.Table[NMossa].NPezzi[3 - Scacc.Table[NMossa - 1].ChiMuove]

        # Se sono arrivato a fine profondità restituisco il risultato stimato
        # dal punto di
        # vista del giocatore che dovrebbe giocare
        if Prof <= 0:
            return -valpr[0]


        # Ordino le mosse secondo le probabilità predette
        if Prof >= 1 and MossePr[0] != None:
            MosseL = Scacc.Table[NMossa].DammiMosseConVuote(Vuote)

            Mosse = list()
            for _ in range(len(MosseL)):
                ProbMax = -1000
                for i in range(len(MosseL)):
                    if (MossePr[ self.CoordBoardToZero(MosseL[i]) ] > ProbMax):
                        ProbMax = MossePr[ self.CoordBoardToZero(MosseL[i]) ]
                        iMax = i
                MossePr[ self.CoordBoardToZero(MosseL[iMax]) ] = -1000
                Mosse.append(MosseL[iMax])

        else:
            Mosse = Scacc.Table[NMossa].DammiMosseConVuote(Vuote)


        if Prof >= 1 and Scacc.MossaFinale-(NMossa+1) > 8:
            # Predico tutte le mosse delle sottoscacchiere di ogni mossa
            # perche' predire piu' scacchiere contemporaneamente e' molto piu'
            # veloce rispetto ad una sola
            boards = list()
            komi = list()
            for i in range(len(Mosse)):
                # Ripristino NMossa che è stato incrementato da FaiMossa
                Scacc.NMossa = NMossa
                Scacc.FaiMossa(Mosse[i], True)
                board = Scacc.DammiBoardDaPredire()
                boards.append(board)
                komi.append([float(0)])
            
            mossep, valp, ap, bp = self.NeuralNet.predici_board(boards, komi)
        else:
            mossep = [0] * 40
            valp = [0] * 40

        for i in range(len(Mosse)):

            # Ripristino NMossa che è stato incrementato da FaiMossa
            Scacc.NMossa = NMossa
            Scacc.FaiMossa(Mosse[i], True)
            self.Calcolate += 1

            Vp = Vuote.index(Mosse[i])
            Vuote.remove(Mosse[i])

            # Se il giocatore successivo sono sempre io allora devo prendere il
            # risultato dal mio punto di vista
            if Scacc.Table[NMossa].ChiMuove != Scacc.Table[NMossa + 1].ChiMuove:
                Val = self.Valuta_Mossa(Scacc, Vuote, NMossa + 1, Prof - 1, -Beta, -Alpha, mossep[i], valp[i])
            else:
                Val = - self.Valuta_Mossa(Scacc, Vuote, NMossa + 1, Prof - 1, Alpha, Beta, mossep[i], valp[i])

            Vuote.insert(Vp, Mosse[i])

            if Val >= Beta:
                return -Val

            if Val > Alpha:
                Alpha = Val

        return -Alpha



    def Valuta_Finale(self, Scacc: SCACC, Vuote, NMossa, Alpha, Beta):

        # Se è finita la partita ritorno la differenza pedine dal punto di
        # vista del giocatore precedente
        if Scacc.Table[NMossa].ChiMuove == 0:
            return Scacc.Table[NMossa].NPezzi[Scacc.Table[NMossa - 1].ChiMuove] - Scacc.Table[NMossa].NPezzi[3 - Scacc.Table[NMossa - 1].ChiMuove]


        Mosse = Scacc.Table[NMossa].DammiMosseConVuote(Vuote)
        
        for i in range(len(Mosse)):

            # Ripristino NMossa che è stato incrementato da FaiMossa
            Scacc.NMossa = NMossa
            Scacc.FaiMossa(Mosse[i], True)
            self.Calcolate += 1

            if (NMossa < Scacc.MossaFinale-1):
                Vp = Vuote.index(Mosse[i])
                Vuote.remove(Mosse[i])

            # Se il giocatore successivo sono sempre io allora devo prendere il
            # risultato dal mio punto di vista
            if Scacc.Table[NMossa].ChiMuove != Scacc.Table[NMossa + 1].ChiMuove:
                Val = self.Valuta_Finale(Scacc, Vuote, NMossa + 1, -Beta, -Alpha)
            else:
                Val = - self.Valuta_Finale(Scacc, Vuote, NMossa + 1, Alpha, Beta)

            if (NMossa < Scacc.MossaFinale-1):
                Vuote.insert(Vp, Mosse[i])

            if Val >= Beta:
                return -Val

            if Val > Alpha:
                Alpha = Val

        return -Alpha





