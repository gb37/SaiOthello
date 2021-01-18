
import numpy as np



Size = 6
bSize = Size + 2
MossaFinale = (Size*Size)-3
VuoteFinalePerfetto = 8
MossaFinalePerfetto = ((Size*Size)-3) - VuoteFinalePerfetto

dir = [(-1,-1), (0,-1), (+1,-1), (-1,0), (+1,0), (-1,+1), (0,+1), (+1,+1)]

OrdV88Prima = [20,
                1,   # Angoli
                4,
               19,
               12,
                3,
                2 ]  # Caselle C  
                     # Tutte le altre caselle vengono dopo le Caselle C e prima delle Caselle D

OrdV88Dopo = [11,  # Caselle D  
              10 ] # Caselle X

ListaVuoteOrdinata = []
Dirs = [0] * 8


def InizializzaDirEVuote(initSize, initVuotePerfetto):
    
    global Size
    Size = initSize
    global bSize
    bSize = Size + 2
    
    global VuoteFinalePerfetto
    VuoteFinalePerfetto = initVuotePerfetto
    global MossaFinale
    MossaFinale = (Size*Size)-3
    global MossaFinalePerfetto
    MossaFinalePerfetto = ((Size*Size)-3) - VuoteFinalePerfetto

    global Dirs
    global ListaVuoteOrdinata

    for i in range(0, 8):
        Dirs[i] = dir[i][0] * bSize + dir[i][1]
    
    ListaVuoteDopo = []

    # Inizio ad inserire tutte le vuote secondo l'ordine del vettore
    for i in range(0, len(OrdV88Prima)):
        x88 = (OrdV88Prima[i] - 1) % 8 + 1
        y88 = int((OrdV88Prima[i] - 1) / 8) + 1
        #Controllo che questa casella esista con l'attuale grandezza della scacchiera
        if x88 <= Size/2 and y88 <= Size/2:
            
            ListaVuoteOrdinata.append(x88 + y88 * bSize)                           # in alto a sinistra
            ListaVuoteOrdinata.append((bSize-1)-x88 + y88 * bSize)                 # in alto a destra
            ListaVuoteOrdinata.append(x88 + ((bSize-1)-y88) * bSize)               # in basso a sinistra
            ListaVuoteOrdinata.append((bSize-1)-x88 + ((bSize-1)-y88) * bSize)     # in basso a destra
            if x88 != y88:
                #inverto x ed y per la simmetria attorno alla diagonale
                k88 = x88
                x88 = y88
                y88 = k88
            
                ListaVuoteOrdinata.append(x88 + y88 * bSize)                         # in alto a sinistra
                ListaVuoteOrdinata.append((bSize-1)-x88 + y88 * bSize)               # in alto a destra
                ListaVuoteOrdinata.append(x88 + ((bSize-1)-y88) * bSize)             # in basso a sinistra
                ListaVuoteOrdinata.append((bSize-1)-x88 + ((bSize-1)-y88) * bSize)   # in basso a destra


    # Questa è la lista delle vuote che dovrebbero essere inserite solo alla fine
    for i in range(0, len(OrdV88Dopo)):
        x88 = (OrdV88Dopo[i] - 1) % 8 + 1
        y88 = int((OrdV88Dopo[i] - 1) / 8) + 1
        #Controllo che questa casella esista con l'attuale grandezza della scacchiera
        if x88 <= Size/2 and y88 <= Size/2:
            
            ListaVuoteDopo.append(x88 + y88 * bSize)                        # in alto a sinistra
            ListaVuoteDopo.append((bSize-1)-x88 + y88 * bSize)              # in alto a destra
            ListaVuoteDopo.append(x88 + ((bSize-1)-y88) * bSize)            # in basso a sinistra
            ListaVuoteDopo.append((bSize-1)-x88 + ((bSize-1)-y88) * bSize)  # in basso a destra
            if x88 != y88:
                #inverto x ed y per la simmetria attorno alla diagonale
                k88 = x88
                x88 = y88
                y88 = k88
            
                ListaVuoteDopo.append(x88 + y88 * bSize)                        # in alto a sinistra
                ListaVuoteDopo.append((bSize-1)-x88 + y88 * bSize)              # in alto a destra
                ListaVuoteDopo.append(x88 + ((bSize-1)-y88) * bSize)            # in basso a sinistra
                ListaVuoteDopo.append((bSize-1)-x88 + ((bSize-1)-y88) * bSize)  # in basso a destra


    # Inserisco le vuote solo se non sono quelle che vanno per ultime e non sono già state inserite
    for y in range(1, bSize - 1):
        for x in range(1, bSize - 1):
            try:
                ListaVuoteDopo.index(x+ y* bSize)
            except:
                try:
                    ListaVuoteOrdinata.index(x+ y* bSize)
                except:
                    ListaVuoteOrdinata.append(x+ y* bSize)


    # Inserisco tutte le vuote secondo l'ordine delle ultime
    for i in range(0, len(OrdV88Dopo)):
        x88 = (OrdV88Dopo[i] - 1) % 8 + 1
        y88 = int((OrdV88Dopo[i] - 1) / 8) + 1
        #Controllo che questa casella esista con l'attuale grandezza della scacchiera
        if x88 <= Size/2 and y88 <= Size/2:
            
            ListaVuoteOrdinata.append(x88 + y88 * bSize)                               # in alto a sinistra
            ListaVuoteOrdinata.append((bSize-1)-x88 + y88 * bSize)                 # in alto a destra
            ListaVuoteOrdinata.append(x88 + ((bSize-1)-y88) * bSize)               # in basso a sinistra
            ListaVuoteOrdinata.append((bSize-1)-x88 + ((bSize-1)-y88) * bSize) # in basso a destra
            if x88 != y88:
                #inverto x ed y per la simmetria attorno alla diagonale
                k88 = x88
                x88 = y88
                y88 = k88
            
                ListaVuoteOrdinata.append(x88 + y88 * bSize)                               # in alto a sinistra
                ListaVuoteOrdinata.append((bSize-1)-x88 + y88 * bSize)                 # in alto a destra
                ListaVuoteOrdinata.append(x88 + ((bSize-1)-y88) * bSize)               # in basso a sinistra
                ListaVuoteOrdinata.append((bSize-1)-x88 + ((bSize-1)-y88) * bSize) # in basso a destra


    # Controllo se manca qualche casella Vuota
    for y in range(1, bSize - 1):
        for x in range(1, bSize - 1):
            try:
                ListaVuoteOrdinata.index(x+ y* bSize)
            except:
                ListaVuoteOrdinata.append(x+ y* bSize)



class BOARD:

    def __init__(self):

        self.B = []
        self.NPezzi = []
        self.ChiMuove = 0
        self.NMossa = 0

     
    def Inizializza(self):
        
        # Creo la scacchiera vuota
        self.B = [0] * (bSize * bSize)

        # Metto le 4 pedine iniziali
        self.B[int((bSize - 2) / 2) * bSize + int((bSize - 2) / 2)] = 2
        self.B[(int((bSize - 2) / 2) + 1) * bSize + int((bSize - 2) / 2) + 1] = 2
        self.B[(int((bSize - 2) / 2) + 1) * bSize + int((bSize - 2) / 2)] = 1
        self.B[int((bSize - 2) / 2) * bSize + int((bSize - 2) / 2) + 1] = 1

        self.NPezzi = [0, 2, 2]
        self.ChiMuove = 1

        
        
    # Controlla se la mossa si puo' giocare in una casella sicuramente Vuota
    def ProvaMossa(self, Pos):
                
        Avv = 3 - self.ChiMuove
      
        for dir in Dirs:
            
            tPos = Pos + dir
            
            Ok = False
            while self.B[tPos] == Avv:
                tPos += dir
                Ok = True
 
            if Ok and self.B[tPos] == self.ChiMuove:  
                return True
  
        return False
    
    
    # Controlla se c'e' almeno una mossa per il giocatore scelto
    def TrovaMossa(self, ChiMuove):

        OrgChiMuove = self.ChiMuove
        self.ChiMuove = ChiMuove
        
        for y in range(1, bSize - 1):
            for x in range(1, bSize - 1):
                if self.B[y * bSize + x] == 0:
                    if self.ProvaMossa(y * bSize + x):
                        self.ChiMuove = OrgChiMuove
                        return True
        
        self.ChiMuove = OrgChiMuove
        return False



    # Restituisce la lista delle caselle vuote
    def DammiVuote(self):

        Vuote = []
        
        for i in range(0, len(ListaVuoteOrdinata)):
            Pos = ListaVuoteOrdinata[i] 
            if self.B[Pos] == 0:
                Vuote.append(Pos)
        
        return Vuote

    
    # Restituisce le mosse possibili
    def DammiMosse(self):

        M = []
        
        for i in range(0, len(ListaVuoteOrdinata)):
            Pos = ListaVuoteOrdinata[i] 
            if self.B[Pos] == 0:
                if self.ProvaMossa(Pos):
                    M.append(Pos)
        
        return M


    # Restituisce le mosse possibili avendo già la lista delle Vuote
    def DammiMosseConVuote(self, ListaVuote):

        M = []
        
        for i in range(0, len(ListaVuote)):
            Pos = ListaVuote[i] 
            if self.ProvaMossa(Pos):
                M.append(Pos)
        
        return M

    
    # Esegue uma mossa che sicuramente si puo' giocare
    def EseguiMossa(self, Pos):
      
        Avv = 3 - self.ChiMuove
        p = 0 

        for dir in Dirs:
            
            tPos = Pos + dir

            Ok = False
            while self.B[tPos] == Avv:
                tPos += dir
                Ok = True
 
            if Ok and self.B[tPos] == self.ChiMuove:  
                tPos -= dir
 
                while self.B[tPos] == Avv:
                    self.B[tPos] = self.ChiMuove
                    tPos -= dir
                    p += 1
  
        self.B[Pos] = self.ChiMuove

        self.NPezzi[Avv] -= p
        self.NPezzi[self.ChiMuove] += p + 1
        
        self.ChiMuove = Avv

        
    def DammiBoardDaPredire(self):

        # Dichiaro il piano bidimensionale contenente nelle caselle
        # pedine di ChiMuove, pedine Avversario
        MtChiMuove = np.zeros(shape=(Size, Size))

        # Riempio i primi 2 piani con le pedine
        for y in range(1, Size + 1):
            for x in range(1, Size + 1):
                if (self.B[y * bSize + x] == self.ChiMuove):
                    MtChiMuove[y - 1][x - 1] = 1.0
                elif (self.B[y * bSize + x] == 3 - self.ChiMuove):
                    MtChiMuove[y - 1][x - 1] = -1.0

        return MtChiMuove
