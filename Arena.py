
import numpy as np
import random
from Utils import *
import Board
import os
import json
from NeuralNet import NEURALNET
from Scacc import SCACC
from Eval import EVAL
from UCTS import *
import pickle
import bz2
import matplotlib.pyplot as plt




class ARENA:

    def __init__(self, TotalGames=100, numMCTSSims=200, GPUBatch=32, ArenaPath = None, GiocatePath = None, RatingsPath = None, StatePath = None, Verbose = 0):

        self.TotalGames = TotalGames
        self.numMCTSSims = numMCTSSims
        self.GPUBatch = GPUBatch
        self.ArenaPath = ArenaPath
        self.GiocatePath = GiocatePath
        self.RatingsPath = RatingsPath
        self.StatePath = StatePath
        self.Verbose = Verbose

        self.Ratings = {}

        if self.RatingsPath != None and os.path.exists(self.RatingsPath):
            with bz2.BZ2File(self.RatingsPath, "rb") as fp:   # Unpickling
                self.Ratings = pickle.load(fp)

        self.TempoP = datetime.datetime.now()


        Lx = list(self.Ratings.keys())
        Ly = list(self.Ratings.values())
        if len(Lx) > 2:
            ListX = [x for x, _ in sorted(zip(Lx, Ly))]
            ListY = [y for _, y in sorted(zip(Lx, Ly))]
            plt.plot(ListX, ListY)
            plt.show()






    def EseguiIncontro(self, Net1, Net2, Net1Gen, Net2Gen):

        TotalGames = self.TotalGames

        XOT = 1
        RandXOT = 1
        NumMosseRand = 5

        ListXOTRandom = np.random.randint(10780, size=TotalGames)

        Vinte = 0
        Pari = 0
        Perse = 0
        Komi = 0

        Giocatore = 1
        NumGame = 0
        NumGameXOT = 0
        TotDiffPed = 0

        ApertureGiocate = list()
        Risultati = list()

        Scacc = SCACC(Board.Size)
        Eval = EVAL(Board.Size, Net1)


        while NumGame < TotalGames:

            NumGame += 1
            NumGameXOT += 1


            if XOT > 0 and Board.Size >= 8:
                Seq = LeggiXOT("large-xot.txt", ListXOTRandom[NumGameXOT-1]+1 if RandXOT == 1 else NumGameXOT+XOT)
                Scacc.ImpostaSequenza(Seq)
            else:
                if len(ApertureGiocate) >= NumGameXOT:
                    Seq = ApertureGiocate[NumGameXOT-1]
                    Scacc.ImpostaSequenza(Seq)
                else:
                    ApertureGiocate.append("")


            tsNet1 = UCTS(Net1, Scacc.Table[Scacc.NMossa], self.numMCTSSims, self.GPUBatch)
            tsNet2 = UCTS(Net2, Scacc.Table[Scacc.NMossa], self.numMCTSSims, self.GPUBatch)

            while Scacc.Table[Scacc.NMossa].ChiMuove != 0 and Scacc.NMossa < Board.MossaFinalePerfetto:
                #Scacc.Visualizza()

                if Scacc.NMossa <= NumMosseRand:
                    MosseL = Scacc.Table[Scacc.NMossa].DammiMosse()
                    iScelta = random.randint(0, len(MosseL)-1)
                    Pos = MosseL[iScelta]

                    ApertureGiocate[NumGame-1] += PosToString(Pos)
                else:
                    if Scacc.Table[Scacc.NMossa].ChiMuove == Giocatore:
                        Pos, BestPos, Val, Alpha, probs = tsNet1.suggest_move(None, initCPUct = 1.0, SceltaTemp = 0.0, ProbsTemp = 1.0, Komi = 0, VisVal = False)
                    else:
                        Pos, BestPos, Val, Alpha, probs = tsNet2.suggest_move(None, initCPUct = 1.0, SceltaTemp = 0.0, ProbsTemp = 1.0, Komi = 0, VisVal = False)


                if Pos%(Scacc.Size+2) >= 1 and Pos%(Scacc.Size+2) <= Scacc.Size and int(Pos/(Scacc.Size+2)) >= 1 and int(Pos/(Scacc.Size+2)) <= Scacc.Size:
                    if Scacc.FaiMossa(Pos):
                        tsNet1.PlayRootMove(Pos)
                        tsNet2.PlayRootMove(Pos)
                    
            
            #Scacc.Visualizza()
            
            if Scacc.Table[Scacc.NMossa].ChiMuove == 0:
                Val = Scacc.Table[Scacc.NMossa].NPezzi[Giocatore] - Scacc.Table[Scacc.NMossa].NPezzi[3-Giocatore]
            else:
                Mossa, Val = Eval.Valuta_Scacc(Scacc, 1)
                if Scacc.Table[Scacc.NMossa].ChiMuove != Giocatore:
                    Val = -Val


            DiffPed = Val
            TotDiffPed += DiffPed
            MedDiffPed = TotDiffPed / NumGame

            Risultati.append(DiffPed)

            if DiffPed > 0:
                Vinte = Vinte + 1
            else:
                if DiffPed == 0:
                    Pari = Pari + 1
                else:
                    Perse = Perse + 1

            StrRis = "  [%+3d" % Risultati[NumGameXOT-1]
            if NumGameXOT == NumGame:
                StrRis += "    ]"
            else:
                StrRis += " %+3d]" % DiffPed


            if XOT > 0 and Board.Size >= 8:
                StrLineaGiocata = "XOT:%5d" % (ListXOTRandom[NumGameXOT-1]+1)
            else:
                StrLineaGiocata = ApertureGiocate[NumGameXOT-1]


            Sequenza = ""
            for i in range(1, Scacc.NMossa):
                Sequenza += PosToString(Scacc.Seq[i])
            Sequenza += "  (%+d)" % DiffPed
            Sequenza += "\n"

            Sequenza += "Net %04d" % Net1Gen + " [%d]" % (Giocatore) + " vs %04d" % Net2Gen + "  P:%3d" % NumGame + " (%3d" % Vinte + " %3d" % Pari + (" %3d)" % Perse) + StrRis + " %+6.2f" % MedDiffPed + "  " + StrLineaGiocata
            Sequenza += "\n\n"

            if self.GiocatePath != None:
                with open(self.GiocatePath, 'a') as file:
                    file.write(Sequenza)

            TempoA = datetime.datetime.now()
            TempoT = TempoA - self.TempoP            
            TTrascorso = "   [%02dh:%02dm:%02ds]" % (TempoT.total_seconds() // 3600, (TempoT.total_seconds() // 60) % 60, TempoT.total_seconds() % 60)

            print("Net %04d" % Net1Gen + " [%d]" % (Giocatore) + " vs %04d" % Net2Gen + "  P:%3d" % NumGame + " (%3d" % Vinte + " %3d" % Pari + (" %3d)" % Perse) + StrRis + " %+6.2f" % MedDiffPed + "  " + StrLineaGiocata, TTrascorso)


            Scacc.NMossa = 1

            if (NumGame == TotalGames/2):
                NumGameXOT = 0
                Giocatore = 3-Giocatore
            

        PercNet1Win = (Vinte+Pari*0.5) / NumGame

        return PercNet1Win, MedDiffPed





    def ValutaNet(self, NetGen, NetPath):

        LivelliTest = [-3, -8, -16, -32, -64, -128]


        GamesWinNet1 = 0 
        GamesLossNet1 = 0 
        TotRatingsAvv = 0
        GamesPlayed = 0

        Net1Path = NetPath.replace(".keras", ("_Gen%d.keras" % NetGen))
        if not os.path.exists(Net1Path):
            return

        Net1 = NEURALNET(Board.Size, Net1Path)

        LimiteBasso = False
        for i in range(len(LivelliTest)):

            # Controllo che non ci sia già una nuova Generazione di Rete da Testare
            NCicli = 0 
            if self.StatePath != None and os.path.exists(self.StatePath):
                with open(self.StatePath) as json_file:
                    data = json.load(json_file)
                    for p in data['Stati']:
                        NCicli = p['NCicli']

            if NetGen < NCicli:
                break



            AvvGen = NetGen + LivelliTest[i]

            # se vado sotto 1 uso 1 ma solo una volta
            if AvvGen <= 1:
                if LimiteBasso:
                    break

                LimiteBasso = True
                AvvGen = 1


            Net2Path = NetPath.replace(".keras", ("_Gen%d.keras" % AvvGen))
            if not os.path.exists(Net2Path):
                break

            Net2 = NEURALNET(Board.Size, Net2Path)

            PercVitt, RisMed = self.EseguiIncontro(Net1, Net2, NetGen, AvvGen)

            # Se ho vinto il 100% delle partite vuol dire che non riesco a capire quanto sono migliore del mio avversario
            if PercVitt > 0.99:
                break

            GamesWinNet1 += PercVitt * self.TotalGames
            GamesLossNet1 += (1.0 - PercVitt) * self.TotalGames
            GamesPlayed += self.TotalGames


            # Se non è stato ancora assegnato un Rating al mio avversario cerco quello prima più vicino altrimenti gli assegno un rating iniziale di 0 punti
            if AvvGen not in self.Ratings:
                self.Ratings[AvvGen] = 0

                k = AvvGen-1
                while k > 1 and k not in self.Ratings:
                    k -= 1

                if k > 1 and k in self.Ratings:
                    self.Ratings[AvvGen] = self.Ratings[k]


            TotRatingsAvv += (self.Ratings[AvvGen] * self.TotalGames)

            
            Risultato = ("Net %04d" % NetGen) + (" vs %04d" % AvvGen) + (" : %5.2f%%" % (PercVitt*100.0)) + (" (%+6.2f)" % RisMed) + "\n"
            if self.ArenaPath != None:
                with open(self.ArenaPath, 'a') as file:
                    file.write(Risultato)
            

            TempoA = datetime.datetime.now()
            TempoT = TempoA - self.TempoP            
            TTrascorso = "   [%02dh:%02dm:%02ds]" % (TempoT.total_seconds() // 3600, (TempoT.total_seconds() // 60) % 60, TempoT.total_seconds() % 60)

            print(("Net %04d" % NetGen) + (" vs %04d" % AvvGen) + (" : %5.2f%%" % (PercVitt*100.0)) + (" (%+6.2f)" % RisMed), TTrascorso)

            if GamesPlayed > 0:
                self.Ratings[NetGen] = int((TotRatingsAvv + 400 * (GamesWinNet1 - GamesLossNet1)) / GamesPlayed)

                with bz2.BZ2File(self.RatingsPath, "wb") as fp:   #Pickling
                    pickle.dump(self.Ratings, fp)

                Risultato = ("Net %04d" % NetGen) + (" Rating: %d" % self.Ratings[NetGen]) + "\n"
                if self.ArenaPath != None:
                    with open(self.ArenaPath, 'a') as file:
                        file.write(Risultato)

                print(("Net %04d" % NetGen) + (" Rating: %d" % self.Ratings[NetGen]))





        if GamesPlayed > 0:
            if self.Verbose == 2:
                Lx = list(self.Ratings.keys())
                Ly = list(self.Ratings.values())
                if len(Lx) > 2:
                    ListX = [x for x, _ in sorted(zip(Lx, Ly))]
                    ListY = [y for _, y in sorted(zip(Lx, Ly))]
                    plt.plot(ListX, ListY)
                    plt.show()




