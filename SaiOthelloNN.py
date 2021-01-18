
import numpy as np
import random
import datetime
import time
import sys
import os
import json
from Utils import *
from Board import *
import Board


use_pytorch = 1




class Logger(object):
    def __init__(self, LogPath):
        self.terminal = sys.stdout
        self.log = open(LogPath, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  
        self.log.flush()

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    


def LeggiMossaFile(FilePath):
    
    NonLetto = 1
    while NonLetto == 1:
        try:
            with open(FilePath) as file:
                read_data = file.read()
                file.close()
                
                if len(read_data) < 2:
                    time.sleep(0.2)
                    continue

                os.remove(FilePath)
                NonLetto = 0
        except:
            NonLetto = 1
            time.sleep(0.2)
    
    return read_data
    

def InviaMossaFile(Mossa, FilePath):
    while os.path.exists(FilePath):
        time.sleep(0.2)
    
    with open(FilePath, 'w') as file:
        file.write(Mossa)
        file.close()
               

def PosToString(Pos):
    
    Mossay = int(Pos/Board.bSize)
    Mossax = Pos%Board.bSize
    return chr(65 + Mossax - 1) + str(Mossay)


        
def VisualizzaBoardMosse(Size, board, mosse_pred, ris_reale, mosse, komi, winlose_reale, ris_pred, Beta_pred, winlose_pred):

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
                if mosse[y][x] >= 0.01:
                    RigaP += " %1.2f" % mosse[y][x]
                else:
                    RigaP += "   . "

                if mosse_pred[y][x] >= 0.01:
                    RigaM += " %1.2f" % mosse_pred[y][x]
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

        print("Ris.reale: %+6.2f" % ris_reale, " " * (Size-6)*3, "    Komi: %+1.2f" % komi, "  WLD.reale: %+1.2f" % winlose_reale, " " * (Size-6)*3, "       A: %+1.2f" % ris_pred, "B: %+1.2f" % Beta_pred, " WDL.pred: %+1.2f" % winlose_pred)



def main(argv):

    # SaiOthelloNN
    
    no_cuda = 0

    import SaiOthelloNN
    SaiOthelloNN.use_pytorch = 0

    
    
    Size = 6
    Filters = 144   # Size*Size * 4
    NetDepth = 12   # Filters / 12,8

    NumGames = 200
    EsempiMax = 500000

    LearnRate = 0.0005
    Epochs = 10
    BatchSize = 512
    Verbose = 1
    
    Depth = 4
    VuotePerfetto = 8

    Modo = 3
    AzzeraCicli = 0
    
    NetGen = 1

    # Il Komi dal punto di vista del Nero, vale anche come KomiMax durante il training
    # Komi -4 vuol dire che se il Nero fa -4 pareggia   Val-Komi = Risultato centrato sullo zero
    Komi = 8
    AllGames = 1
    
    TempoMax = 2
    ChiUmano = 2
    GiocoFile = 0
    XOT = 0
    RandXOT = 0

    numMCTSSims = 200
    NThreads = 1
    GPUBatch = 32

    CPUct = 1.0
    SceltaTemp = 2.0
    ProbsTemp = 4.0

    Samples = 20


 
    LogPath = "C:/Temp/"
    NetPath = "C:/Temp/"
    StatePath = "C:/Temp/"
    MemoryPath = "C:/Temp/"
    RatingsPath = "C:/Temp/"
    ArenaPath = "C:/Temp/"
    GiocatePath = "C:/Temp/"
    EsempiPath = "C:/Temp/"

    MossaInPath = "C:/Temp/MossaOther.txt"
    MossaOutPath = "C:/Temp/MossaSaio.txt"

    

    # Analizzo le opzioni dalla riga di comando
    for i in range(len(argv)):
        if argv[i] == "-m":
            Modo = 1
        elif argv[i] == "-UsePytorch" and len(argv) > i + 1:
            SaiOthelloNN.use_pytorch = int(argv[i + 1])
        elif argv[i] == "-NoCuda" and len(argv) > i + 1:
            no_cuda = int(argv[i + 1])
        elif argv[i] == "-t" and len(argv) > i + 1:
            TempoMax = int(argv[i + 1])
        elif argv[i] == "-LearnRate" and len(argv) > i + 1:
            LearnRate = float(argv[i + 1])
        elif argv[i] == "-BatchSize" and len(argv) > i + 1:
            BatchSize = int(argv[i + 1])
        elif argv[i] == "-Epochs" and len(argv) > i + 1:
            Epochs = int(argv[i + 1])
        elif argv[i] == "-NumGames" and len(argv) > i + 1:
            NumGames = int(argv[i + 1])
        elif argv[i] == "-EsempiMax" and len(argv) > i + 1:
            EsempiMax = int(argv[i + 1])
        elif argv[i] == "-NThreads" and len(argv) > i + 1:
            NThreads = int(argv[i + 1])
        elif argv[i] == "-GPUBatch" and len(argv) > i + 1:
            GPUBatch = int(argv[i + 1])
        elif argv[i] == "-NetGen" and len(argv) > i + 1:
            NetGen = int(argv[i + 1])
        elif argv[i] == "-AllGames" and len(argv) > i + 1:
            AllGames = int(argv[i + 1])
        elif argv[i] == "-Size" and len(argv) > i + 1:
            Size = int(argv[i + 1])
        elif argv[i] == "-Filters" and len(argv) > i + 1:
            Filters = int(argv[i + 1])
        elif argv[i] == "-NetDepth" and len(argv) > i + 1:
            NetDepth = int(argv[i + 1])
        elif argv[i] == "-Depth" and len(argv) > i + 1:
            Depth = int(argv[i + 1])
        elif argv[i] == "-VuotePerfetto" and len(argv) > i + 1:
            VuotePerfetto = int(argv[i + 1])
        elif argv[i] == "-ChiUmano" and len(argv) > i + 1:
            ChiUmano = int(argv[i + 1])
        elif argv[i] == "-Modo" and len(argv) > i + 1:
            Modo = int(argv[i + 1])
        elif argv[i] == "-Komi" and len(argv) > i + 1:
            Komi = int(argv[i + 1])
        elif argv[i] == "-Verbose" and len(argv) > i + 1:
            Verbose = int(argv[i + 1])
        elif argv[i] == "-GiocoFile" and len(argv) > i + 1:
            GiocoFile = int(argv[i + 1])
        elif argv[i] == "-XOT" and len(argv) > i + 1:
            XOT = int(argv[i + 1])
        elif argv[i] == "-RandXOT" and len(argv) > i + 1:
            RandXOT = int(argv[i + 1])
        elif argv[i] == "-MCTS" and len(argv) > i + 1:
            numMCTSSims = int(argv[i + 1])
        elif argv[i] == "-CPUct" and len(argv) > i + 1:
            CPUct = float(argv[i + 1])
        elif argv[i] == "-SceltaTemp" and len(argv) > i + 1:
            SceltaTemp = float(argv[i + 1])
        elif argv[i] == "-ProbsTemp" and len(argv) > i + 1:
            ProbsTemp = float(argv[i + 1])
        elif argv[i] == "-Samples" and len(argv) > i + 1:
            Samples = int(argv[i + 1])
        elif argv[i] == "-AzzeraCicli" and len(argv) > i + 1:
            AzzeraCicli = int(argv[i + 1])
        elif argv[i] == "-Colab":
            NetPath = "/content/drive/My Drive/SaiOthelloNN/"
            LogPath = "/content/drive/My Drive/SaiOthelloNN/"
            StatePath = "/content/drive/My Drive/SaiOthelloNN/"
            MemoryPath = "/content/drive/My Drive/SaiOthelloNN/"
            RatingsPath = "/content/drive/My Drive/SaiOthelloNN/"
            ArenaPath = "/content/drive/My Drive/SaiOthelloNN/"
            GiocatePath = "/content/drive/My Drive/SaiOthelloNN/"
            EsempiPath = "/content/drive/My Drive/SaiOthelloNN/"
        elif argv[i] == "-Colab1":
            NetPath = "/content/drive/My Drive/SaiOthelloNN/Session1/"
            LogPath = "/content/drive/My Drive/SaiOthelloNN/Session1/"
            StatePath = "/content/drive/My Drive/SaiOthelloNN/Session1/"
            MemoryPath = "/content/drive/My Drive/SaiOthelloNN/Session1/"
            RatingsPath = "/content/drive/My Drive/SaiOthelloNN/Session1/"
            ArenaPath = "/content/drive/My Drive/SaiOthelloNN/Session1/"
            GiocatePath = "/content/drive/My Drive/SaiOthelloNN/Session1/"
            EsempiPath = "/content/drive/My Drive/SaiOthelloNN/"
        elif argv[i] == "-Colab2":
            NetPath = "/content/drive/My Drive/SaiOthelloNN/Session2/"
            LogPath = "/content/drive/My Drive/SaiOthelloNN/Session2/"
            StatePath = "/content/drive/My Drive/SaiOthelloNN/Session2/"
            MemoryPath = "/content/drive/My Drive/SaiOthelloNN/Session2/"
            RatingsPath = "/content/drive/My Drive/SaiOthelloNN/Session2/"
            ArenaPath = "/content/drive/My Drive/SaiOthelloNN/Session2/"
            GiocatePath = "/content/drive/My Drive/SaiOthelloNN/Session2/"
            EsempiPath = "/content/drive/My Drive/SaiOthelloNN/"
        elif argv[i] == "-Colab3":
            NetPath = "/content/drive/My Drive/SaiOthelloNN/Session3/"
            LogPath = "/content/drive/My Drive/SaiOthelloNN/Session3/"
            StatePath = "/content/drive/My Drive/SaiOthelloNN/Session3/"
            MemoryPath = "/content/drive/My Drive/SaiOthelloNN/Session3/"
            RatingsPath = "/content/drive/My Drive/SaiOthelloNN/Session3/"
            ArenaPath = "/content/drive/My Drive/SaiOthelloNN/Session3/"
            GiocatePath = "/content/drive/My Drive/SaiOthelloNN/Session3/"
            EsempiPath = "/content/drive/My Drive/SaiOthelloNN/"


    #sys.stdout = Logger(LogPath)

    


    
    SuffissoSize = "_" + str(Size) + "x" + str(Size)
    SuffissoNet = "_" + str(Filters) + "x" + str(NetDepth)



    if no_cuda == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if SaiOthelloNN.use_pytorch == 1:
        SuffissoNNLib = ".pytorch"

        import torch as pt
        pt.backends.cudnn.benchmark = True
    else:
        SuffissoNNLib = ".keras"

        import tensorflow as tf
        import tensorflow.keras as keras
        from tensorflow.keras.mixed_precision import experimental as mixed_precision
        from tensorflow.keras import layers, models

        if no_cuda != 1:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)


    NetPath += "SaiOthelloNN" + SuffissoSize + SuffissoNet + SuffissoNNLib
    StatePath += "SaiOthelloNN" + SuffissoSize + SuffissoNet + SuffissoNNLib + ".state"
    MemoryPath += "SaiOthelloNN" + SuffissoSize + SuffissoNet + SuffissoNNLib + ".memory"
    RatingsPath += "SaiOthelloNN" + SuffissoSize + SuffissoNet + SuffissoNNLib + ".ratings"
    ArenaPath += "SaiOthelloNN" + SuffissoSize + SuffissoNet + SuffissoNNLib + ".arena.txt"
    GiocatePath += "SaiOthelloNN" + SuffissoSize + SuffissoNet + SuffissoNNLib + ".giocate.txt"
    LogPath += "SaiOthelloNN" + SuffissoSize + SuffissoNet + SuffissoNNLib + ".log"

    EsempiPath += "EsempiUCTS" + SuffissoSize + ".txt"



    print("Modo:", Modo)
    print("ChiUmano:", ChiUmano)
    print("NoCuda:", no_cuda)
    print("UsePytorch:", SaiOthelloNN.use_pytorch)
    print("XOT:", XOT)
    print("Size:", Size)
    print("Filters:", Filters)
    print("NetDepth:", NetDepth)
    print("LearnRate:", LearnRate)
    print("Epochs:", Epochs)
    print("BatchSize:", BatchSize)
    print("Verbose:", Verbose)
    print("NumGames:", NumGames)
    print("EsempiMax:", EsempiMax)
    print("Samples:", Samples)
    print("NThreads:", NThreads)
    print("GPUBatch:", GPUBatch)
    print("Depth:", Depth)
    print("VuotePerfetto:", VuotePerfetto)
    print("MCTS:", numMCTSSims)
    print("CPUct:", CPUct)
    print("SceltaTemp:", SceltaTemp)
    print("ProbsTemp:", ProbsTemp)
    print("NetGen:", NetGen)
    print("Komi:", Komi)
    print("AllGames:", AllGames)
    print("AzzeraCicli:", AzzeraCicli)




    if Size < 6:
        print("Size", Size, "NON supportata")
        return

    InizializzaDirEVuote(Size, VuotePerfetto)
    DaVuote = Board.MossaFinale - 2



    if ChiUmano == 1:
        MossaOutPath = "C:/Temp/MossaOther.txt"
        MossaInPath = "C:/Temp/MossaSaio.txt"



    if GiocoFile == 1:
        try:
            os.remove(MossaOutPath)
        except:
            xx=0





    from Scacc import SCACC
    from Games import GAMES
    from Eval import EVAL
    from Train import TRAIN
    from UCTS import UCTS
    from Arena import ARENA


    if SaiOthelloNN.use_pytorch == 1:
        from NeuralNet import NEURALNETPytorch
    else:
        from NeuralNet import NEURALNET

    



    Scacc = SCACC(Size)
    Scacc.Visualizza()
   
    
    # Modalità gioco manuale
    if Modo == 1:
    
        if SaiOthelloNN.use_pytorch == 1:
            NeuralNet = NEURALNETPytorch(Size, net_file=NetPath, learn_rate=LearnRate, l2a=0, l2k=0, Filters=Filters, NetDepth=NetDepth)
        else:
            NeuralNet = NEURALNET(Size, net_file=NetPath, learn_rate=LearnRate, l2a=0, l2k=0, Filters=Filters, NetDepth=NetDepth)


        Eval = EVAL(Size, NeuralNet)


        TotalGames = 100

        ApertureGiocate = list()
        RandXOT = 1
        NumMosseRand = 5

        ListXOTRandom = np.random.randint(10780, size=TotalGames)

        Vinte = 0
        Pari = 0
        Perse = 0
        SecondaMeta = 0

        Giocatore = (3-ChiUmano)
        NumGame = 0
        NumGameXOT = 0
        TotDiffPed = 0

        while NumGame < TotalGames:

            NumGame += 1
            NumGameXOT += 1


            if Giocatore == 1 and XOT > 0 and Board.Size >= 8:
                Seq = LeggiXOT("large-xot.txt", ListXOTRandom[NumGameXOT-1]+1 if RandXOT == 1 else NumGameXOT+XOT)

                # Se Size >= 10 modificare Seq in modo da portare le mosse da Pos88 a Pos100 

                Scacc.ImpostaSequenza(Seq)

                if GiocoFile == 1 and SecondaMeta == 0:
                    InviaMossaFile(Seq, MossaOutPath)
            else:
                if len(ApertureGiocate) >= NumGameXOT:
                    Seq = ApertureGiocate[NumGameXOT-1]
                    Scacc.ImpostaSequenza(Seq)
                else:
                    ApertureGiocate.append("")
            
            Scacc.Visualizza()
                
                
            #Scacc.ImpostaSequenza("C4C3D3C5B2E2C2E3C6B6D6E6B5D2F3F2F4G5E1D1G4F5F6F1C1")



            ts = UCTS(NeuralNet, Scacc.Table[Scacc.NMossa], numMCTSSims, GPUBatch)

            while Scacc.Table[Scacc.NMossa].ChiMuove != 0:
                Scacc.Visualizza()

                if Giocatore == 1 and XOT > 0 and Scacc.NMossa <= NumMosseRand:
                    MosseL = Scacc.Table[Scacc.NMossa].DammiMosse()
                    iScelta = random.randint(0, len(MosseL)-1)
                    Pos = MosseL[iScelta]
                    ApertureGiocate[NumGame-1] += PosToString(Pos)

                    if GiocoFile == 1 and Scacc.NMossa == NumMosseRand:
                        InviaMossaFile(ApertureGiocate[NumGame-1], MossaOutPath)

                else:
                    
                    if Scacc.Table[Scacc.NMossa].ChiMuove == ChiUmano:
                        
                        if GiocoFile == 1:
                            Mossa = LeggiMossaFile(MossaInPath).upper()
                        else:                       
                            ts.root.VisDatiNodo()

                            Mossa = input("Mossa: ").upper()

                        Pos = 0



                        if len(Mossa) == 2:
                            x = ord(Mossa[0]) - ord('A') + 1
                            y = ord(Mossa[1]) - ord('1') + 1
                            Pos = y * (Scacc.Size+2) + x

                        # Se mi viene mandata una sequenza alla prima mossa allora la imposto come iniziale
                        if Scacc.NMossa == 1 and len(Mossa) >= 4:
                            Scacc.ImpostaSequenza(Mossa)
                            ApertureGiocate[NumGame-1] = Mossa
                            ts = UCTS(NeuralNet, Scacc.Table[Scacc.NMossa], numMCTSSims, GPUBatch)

                    else:

                        TempoP = datetime.datetime.now()

                        if numMCTSSims > 0:
                            tKomi = Komi
                            if Scacc.Table[Scacc.NMossa].ChiMuove == 2:
                                tKomi = -Komi
                            
                            ts.root.VisDatiNodo()
                            Pos, BestPos, Val, Alpha, probs = ts.suggest_move(None, CPUct, SceltaTemp, ProbsTemp, Komi, True)
                            board = Scacc.DammiBoardDaPredire()
                            probs_pred, winlose, alpha, beta =  NeuralNet.predici_board(board, tKomi)
                            Scacc.VisualizzaBoardMosse(np.reshape(probs_pred, (Size, Size)), np.reshape(probs, (Size, Size)), tKomi, alpha, beta, winlose)
                            ts.root.VisDatiNodo()

                        else:
                            Prof = 2 + (Depth % 2)
                            while Prof <= Depth:
                                Pos, Val = Eval.Valuta_Scacc(Scacc, Prof, 0)

                                TempoA = datetime.datetime.now()
                                TempoT = TempoA - TempoP            

                                print("Prof:%2d" % Prof, (chr(65 + Pos%(Scacc.Size+2) - 1) + str(int(Pos/(Scacc.Size+2))) + ":%+1.2f " % Val), "Sec:", TempoT.total_seconds(), "Calc:", Eval.Calcolate, "M/s:", Eval.Calcolate/TempoT.total_seconds())

                                Prof += 2

                        if GiocoFile == 1:
                            InviaMossaFile(chr(65 + Pos%(Scacc.Size+2) - 1) + str(int(Pos/(Scacc.Size+2))), MossaOutPath)
            

                if Pos%(Scacc.Size+2) >= 1 and Pos%(Scacc.Size+2) <= Scacc.Size and int(Pos/(Scacc.Size+2)) >= 1 and int(Pos/(Scacc.Size+2)) <= Scacc.Size:
                    if Scacc.FaiMossa(Pos):
                        ts.PlayRootMove(Pos)
                    
            
            Scacc.Visualizza()
            


            DiffPed = Scacc.Table[Scacc.NMossa].NPezzi[3-ChiUmano] - Scacc.Table[Scacc.NMossa].NPezzi[ChiUmano]
            TotDiffPed += DiffPed
            MedDiffPed = TotDiffPed / NumGame

            if DiffPed > 0:
                Vinte = Vinte + 1
            else:
                if DiffPed == 0:
                    Pari = Pari + 1
                else:
                    Perse = Perse + 1

            Sequenza = ""
            for i in range(1, Scacc.NMossa):
                Sequenza += PosToString(Scacc.Seq[i])
            Sequenza += " " + "[%2d" % Scacc.Table[Scacc.NMossa].NPezzi[1] + "-%2d]" % Scacc.Table[Scacc.NMossa].NPezzi[2]
            Sequenza += "\n"

            Sequenza += "Giocatore%d" % (Giocatore) + "  Partite: %2d" % NumGame + "  Vinte: %2d" % Vinte + " Pari: %2d" % Pari + " Perse: %2d" % Perse + "  DiffPed: %+1.2f" % MedDiffPed
            Sequenza += "\n\n"

            with open(GiocatePath, 'a') as file:
                file.write(Sequenza)

            print("Partite: %2d" % NumGame + "  Vinte: %2d" % Vinte + " Pari: %2d" % Pari + " Perse: %2d" % Perse, " DiffPed: %+1.2f" % MedDiffPed)

            Scacc.NMossa = 1

            if (NumGame == TotalGames/2):
                NumGameXOT = 0
                SecondaMeta = 1
                ChiUmano = 3-ChiUmano
            





    # Modalità Test Generazione Nets
    elif Modo == 2:
            
        Arena = ARENA(TotalGames = 100, numMCTSSims = numMCTSSims, GPUBatch = GPUBatch, ArenaPath = ArenaPath, GiocatePath = GiocatePath, RatingsPath = RatingsPath, StatePath = StatePath, Verbose = Verbose)

        LastNCicli = 0
        while True:
            NCicli = 0 
            if StatePath != None and os.path.exists(StatePath):
                with open(StatePath) as json_file:
                    data = json.load(json_file)
                    for p in data['Stati']:
                        NCicli = p['NCicli']
                        
            if NCicli > 1 and NCicli != LastNCicli:
                LastNCicli = NCicli
                Arena.ValutaNet(NCicli, NetPath)

            time.sleep(5)
        




    # Modalità genera partite
    elif Modo == 3:
            
        Train = TRAIN(SaiOthelloNN.use_pytorch, Size, NetPath, StatePath, EsempiPath, LearnRate, BatchSize, Epochs, NumGames, Depth, DaVuote, NetGen, AllGames, Filters, NetDepth, Verbose, Komi, numMCTSSims, LogPath, NThreads, EsempiMax, MemoryPath, CPUct, SceltaTemp, ProbsTemp, Samples, GPUBatch)

        Train.TrainNet(0, AzzeraCicli)




    # Modalità test
    elif Modo == 9:
            
        if SaiOthelloNN.use_pytorch == 1:
            NeuralNet = NEURALNETPytorch(Size, net_file=NetPath, learn_rate=LearnRate, l2a=0, l2k=0, Filters=Filters, NetDepth=NetDepth)
        else:
            NeuralNet = NEURALNET(Size, net_file=NetPath, learn_rate=LearnRate, l2a=0, l2k=0, Filters=Filters, NetDepth=NetDepth)


        Eval = EVAL(Size, NeuralNet)

        Scacc = SCACC(Size)
        ts = UCTS(NeuralNet, None, numMCTSSims, GPUBatch)

        Scacc.Table[24].B = [0, 0, 0, 0, 0, 0, 0, 0,
                             0, 1, 1, 1, 1, 1, 0, 0,
                             0, 1, 2, 1, 1, 1, 1, 0,
                             0, 1, 2, 1, 2, 1, 1, 0,
                             0, 0, 2, 2, 2, 2, 2, 0,
                             0, 0, 2, 0, 1, 1, 2, 0,
                             0, 0, 0, 0, 1, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0]
        Scacc.Table[24].NPezzi = [0, 17, 10]
        Scacc.Table[24].ChiMuove = 2
        Scacc.Table[24].NMossa = 24
        Scacc.NMossa = 24





        Scacc.Table[23].B = [0, 0, 0, 0, 0, 0, 0, 0,
                             0, 1, 1, 0, 1, 1, 0, 0,
                             0, 1, 2, 1, 1, 1, 1, 0,
                             0, 1, 2, 1, 2, 1, 1, 0,
                             0, 0, 2, 2, 2, 2, 2, 0,
                             0, 0, 2, 0, 1, 1, 2, 0,
                             0, 0, 0, 0, 1, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0]
        Scacc.Table[23].NPezzi = [0, 16, 10]
        Scacc.Table[23].ChiMuove = 1
        Scacc.Table[23].NMossa = 23
        Scacc.NMossa = 23



        Scacc.Table[19].B = [0, 0, 0, 0, 0, 0, 0, 0,
                             0, 1, 1, 1, 1, 1, 1, 0,
                             0, 1, 2, 1, 1, 1, 1, 0,
                             0, 1, 2, 1, 2, 1, 1, 0,
                             0, 0, 2, 2, 2, 2, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0]
        Scacc.Table[19].NPezzi = [0, 15, 7]
        Scacc.Table[19].ChiMuove = 1
        Scacc.Table[19].NMossa = 19
        Scacc.NMossa = 19







        Scacc.Visualizza()



        
        tKomi = Komi
        if Scacc.Table[Scacc.NMossa].ChiMuove == 2:
            tKomi = -Komi

        board = Scacc.DammiBoardDaPredire()
        komi = list()
        komi.append(tKomi)
        pedine = list()
        pedine.append(0)
        winlose = list()
        winlose.append(0)


        Pos, BestPos, Val, Alpha, MtMosse = ts.suggest_move(Scacc.Table[Scacc.NMossa], CPUct, SceltaTemp, ProbsTemp, Komi, True)
        ts.root.VisDatiNodo()

        print("Mossa:", PosToString(Pos), "Val: %+1.3f" % Val)

        #Mossax, Mossay, Bestx, Besty, MtMosse = mt.getActionProb(Scacc, numMCTSSims, 0.0, 0, komi[0], 1)
        p_Mosse_batch, p_winlose_batch, p_alpha_batch, p_beta_batch = NeuralNet.predici_board(board, komi)
        VisualizzaBoardMosse(Size, np.reshape(board, [Size, Size]), np.reshape(MtMosse, [Size, Size]), pedine[0], np.reshape(p_Mosse_batch, (Size, Size)), komi[0], winlose[0], p_alpha_batch[0][0], p_beta_batch[0][0], p_winlose_batch[0][0])
        Eval.Valuta_Scacc(Scacc, 2, 1)




        #TempoP = datetime.datetime.now()

        #Cicli = 300
        #for _ in range(Cicli):
        #    Eval.Valuta_Scacc(Scacc, 1, 0)
        
        #TempoA = datetime.datetime.now()
        #TempoT = TempoA - TempoP            

        #print("Calc:", Eval.Calcolate, "Sec:", TempoT.total_seconds(), "Cicli:", Cicli, "C/s:", Cicli/TempoT.total_seconds())



        #Prof = 2

        #while True:
        #    x, y, Val, x2, y2 = Eval.Valuta_Scacc(Scacc, Prof, 1)

        #    TempoA = datetime.datetime.now()
        #    TempoT = TempoA - TempoP            

        #    print("Prof:%2d" % Prof, (chr(65 + x - 1) + str(y) + ":%+1.2f " % Val), "Sec:", TempoT.total_seconds(), "Calc:", Eval.Calcolate, "M/s:", Eval.Calcolate/TempoT.total_seconds())

        #    Prof += 2
        





if __name__ == "__main__":
    main(sys.argv[1:])
