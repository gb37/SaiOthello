
import collections
import math
import datetime

from absl import flags
import numpy as np

from Board import *
from Scacc import *
from Eval import *
import Board


CPUct = 1.0
minparallel_readouts = 32
verbose = 2



def PosToString(Pos):
    
    Mossay = int(Pos/Board.bSize)
    Mossax = Pos%Board.bSize
    return chr(65 + Mossax - 1) + str(Mossay)


def CoordBoardToZero(Pos):

    y = int(Pos / Board.bSize) - 1
    x = Pos % Board.bSize - 1

    return y * Board.Size + x


class DummyNode():
    """A fake node of a MCTS search tree.
    This node is intended to be a placeholder for the root node, which would
    otherwise have no parent node. If all nodes have parents, code becomes
    simpler."""

    def __init__(self):
        self.parent = None
        self.child_N = collections.defaultdict(float)
        self.child_W = collections.defaultdict(float)
        self.child_Wa = collections.defaultdict(float)
        self.child_new = collections.defaultdict(float)
        self.child_Seq = collections.defaultdict(str)

        self.Board = BOARD()
        self.Board.Inizializza()
        self.Board.NMossa = 0
        self.Board.ChiMuove = 2
        # Perchè facendo la prima mossa 0 incrementa il bianco
        self.Board.NPezzi[2] = 1
        # In questo modo farà la prima mossa fittizia del bianco dove già perchè ce la rimetterà
        self.Mosse = [int((Board.bSize - 2) / 2) * Board.bSize + int((Board.bSize - 2) / 2)]

        self.Seq = ""


class UCTSNode():
    """A node of a MCTS search tree.
    A node knows how to compute the action scores of all of its children,
    so that a decision can be made about which move to explore next. Upon
    selecting a move, the children dictionary is updated with a new node.
    position: A go.Position instance
    fmove: A move (coordinate) that led to this position, a flattened coord
            (raw number between 0-N^2, with None a pass)
    parent: A parent MCTSNode.
    """

    def __init__(self, fmove=None, parent=None):
              
        self.parent = parent
        self.fmove = fmove  # move that led to this position, as flattened coords

        # Creo la board del nodo che la contiene
        # Copio la board dal nodo padre a questo e ci faccio la mossa su che la genera
        self.Board = BOARD()
        self.Board.B.extend(parent.Board.B)
        self.Board.NPezzi.extend(parent.Board.NPezzi)
        self.Board.ChiMuove = parent.Board.ChiMuove
        
        Pos = parent.Mosse[fmove]
        self.Board.EseguiMossa(Pos)
        self.Board.NMossa = parent.Board.NMossa+1

        # Controllo che ci sia una mossa disponibile altrimenti o passo o
        # dichiaro finita la partita

        if not self.Board.TrovaMossa(self.Board.ChiMuove):
            if not self.Board.TrovaMossa(3 - self.Board.ChiMuove):
                self.Board.ChiMuove = 0
            else:
                self.Board.ChiMuove = 3 - self.Board.ChiMuove

        # I nodi considerano valori positivi per il Nero
        self.SegnoGiocatore = +1
        if self.Board.ChiMuove == 2:
            self.SegnoGiocatore = -1

        self.Mosse = self.Board.DammiMosse()
        self.NMosse = len(self.Mosse)

        self.Seq = parent.Seq + PosToString(Pos)
        self.BestSeq = ""


        self.is_expanded = False

        # using child_() allows vectorized computation of action score.
        self.child_N = np.zeros([self.NMosse], dtype=np.float32)
        self.child_W = np.zeros([self.NMosse], dtype=np.float32)  
        self.child_Wa = np.zeros([self.NMosse], dtype=np.float32)  
        self.child_prior = np.zeros([self.NMosse], dtype=np.float32)
        # Imposto priorità infinita ai nuovi nodi, in questo modo verranno sempre espansi
        self.child_new = np.ones([self.NMosse], dtype=np.float32) * (1000000.0)
        #self.child_Seq = ["" for _ in range(self.NMosse)]
        self.children = {}  # map of flattened moves to resulting MCTSNode
        self.RisultatoPerfetto = -1000



    def VisDatiNodo(self):

        print()
        print(self.Seq)
        for i in range(self.NMosse):
            print("  P " + PosToString(self.Mosse[i]) + ":%+1.3f" % self.child_prior[i], end='')
        print(end='\n')
        for i in range(self.NMosse):
            print("  Q " + PosToString(self.Mosse[i]) + ":%+1.3f" % self.child_Q[i], end='')
        print(end='\n')
        for i in range(self.NMosse):
            print("  A " + PosToString(self.Mosse[i]) + ":%+6.2f" % self.child_A[i], end='')
        print(end='\n')
        for i in range(self.NMosse):
            print(" AS " + PosToString(self.Mosse[i]) + ":%+1.3f" % self.child_action_score[i], end='')
        print(end='\n')
        for i in range(self.NMosse):
            print("  N " + PosToString(self.Mosse[i]) + ":%6d" % self.child_N[i], end='')
        print(end='\n')



    @property
    def child_action_score(self):
        return (self.child_Q +
                self.child_U + self.child_new)

    @property
    def child_Q(self):
        return self.child_W * self.SegnoGiocatore / (1 + self.child_N)

    @property
    def child_A(self):
        return self.child_Wa * self.SegnoGiocatore / self.child_N

    @property
    def child_U(self):
        return CPUct * math.sqrt(self.N) * self.child_prior / (1 + self.child_N)

    @property
    def Q(self):
        return self.W * self.SegnoGiocatore / (1 + self.N)

    @property
    def A(self):
        return self.Wa * self.SegnoGiocatore / self.N

    @property
    def N(self):
        return self.parent.child_N[self.fmove]

    @property
    def W(self):
        return self.parent.child_W[self.fmove]

    @property
    def Wa(self):
        return self.parent.child_Wa[self.fmove]

    @N.setter
    def N(self, value):
        self.parent.child_N[self.fmove] = value

    @W.setter
    def W(self, value):
        self.parent.child_W[self.fmove] = value

    @Wa.setter
    def Wa(self, value):
        self.parent.child_Wa[self.fmove] = value


    # Setto il fatto che sono stato già espanso in modo da non pesare in child_action_score
    def Set_NoNew(self):
        self.parent.child_new[self.fmove] = 0


    def select_leaf(self):
        current = self
        while True:
            # if a node has never been evaluated, we have no basis to select a child.
            if not current.is_expanded:
                break

            if verbose >= 4:
                current.VisDatiNodo()
           
            best_move = np.argmax(current.child_action_score)
            current = current.maybe_add_child(best_move)

        return current


    def maybe_add_child(self, fcoord):
        """Adds child node for fcoord if it doesn't already exist, and returns it."""
        if fcoord not in self.children:
            
            self.children[fcoord] = UCTSNode(fmove=fcoord, parent=self)

        return self.children[fcoord]


    def add_virtual_loss(self, up_to):
        """Propagate a virtual loss up to the root node.
        Args:
            up_to: The node to propagate until. (Keep track of this! You'll
                need it to reverse the virtual loss later.)
        """
        # This is a "win" for the current node; hence a loss for its parent node
        # who will be deciding whether to investigate this node again.
        
        loss = self.SegnoGiocatore
        self.W += loss

        if self.parent is None or self is up_to:
            return
        self.parent.add_virtual_loss(up_to)


    def revert_virtual_loss(self, up_to):

        revert = -1 * self.SegnoGiocatore
        self.W += revert

        if self.parent is None or self is up_to:
            return
        self.parent.revert_virtual_loss(up_to)


    def incorporate_results(self, move_probs, value, value_a, up_to):

        # If a node was picked multiple times (despite vlosses), we shouldn't
        # expand it more than once.
        if self.is_expanded:
            return
        self.is_expanded = True


        # Si può fare in un'unica operazione vettoriale senza usare il ciclo
        #probs = [x ** (1. / self.TreeTemp) for x in move_probs]
        #probs_sum = sum(probs)
        #move_probs = [x / probs_sum for x in probs]

        # Re-normalize move_probabilities.
        scale = sum(move_probs)
        if scale > 0:
            move_probs *= (1.0 / scale)


        self.child_prior = move_probs
        self.backup_value(value, value_a, up_to=up_to)


    def backup_value(self, value, value_a, up_to):
        """Propagates a value estimation up to the root node.
        Args:
            value: the value to be propagated (1 = black wins, -1 = white wins)
            up_to: the node to propagate until.
        """
        self.N += 1
        self.W += value
        self.Wa += value_a
        #self.parent.child_Seq[self.fmove] = PosToString( self.parent.Mosse[self.fmove] ) + self.BestSeq
        self.Set_NoNew()  # Setto che non sono più un nuovo nodo perchè ho anche backpropagato il mio valore 

        if self.parent is None or self is up_to:
            return

        self.parent.backup_value(value, value_a, up_to)


    def is_done(self):
        """True if the last two moves were Pass or if the position is at a move
        greater than the max depth."""
        return self.Board.ChiMuove == 0 or self.Board.NMossa >= Board.MossaFinalePerfetto


    def best_child(self):
        # Sort by child_N tie break with action score.
        return np.argmax(self.child_N + self.child_action_score / 10000)

    def best_child_Q(self):
        # Sort by child_Q tie break with action score.
        return np.argmax(self.child_Q + self.child_action_score / 10000)

    def inject_noise(self):
        epsilon = 1e-5
        legal_moves = (1 - self.illegal_moves) + epsilon
        a = legal_moves * ([FLAGS.dirichlet_noise_alpha] * (go.N * go.N + 1))
        dirichlet = np.random.dirichlet(a)
        self.child_prior = (self.child_prior * (1 - FLAGS.dirichlet_noise_weight) + dirichlet * FLAGS.dirichlet_noise_weight)







class UCTS():
    def __init__(self, NeuralNet, InitBoard:BOARD = None, num_readouts = 200, parallel_readouts = 32):

        self.NeuralNet = NeuralNet
        self.Eval = EVAL(Board.Size, NeuralNet)

        self.num_readouts = num_readouts

        global minparallel_readouts
        minparallel_readouts = parallel_readouts

        # Il Komi è dal punto di vista del nero, -4 significa che per il Nero -4 è pareggio
        self.Komi = 0
        self.Scacc = SCACC(Board.Size)

        self.root = UCTSNode(0, DummyNode())

        if InitBoard != None:
            self.root.N = 0
            self.root.Board = InitBoard
            self.root.Mosse = self.root.Board.DammiMosse()
            self.root.NMosse = len(self.root.Mosse)
            self.root.is_expanded = False
            self.root.child_N = np.zeros([self.root.NMosse], dtype=np.float32)
            self.root.child_W = np.zeros([self.root.NMosse], dtype=np.float32) 
            self.root.child_Wa = np.zeros([self.root.NMosse], dtype=np.float32) 
            self.root.child_prior = np.zeros([self.root.NMosse], dtype=np.float32)
            self.root.child_new = np.ones([self.root.NMosse], dtype=np.float32) * (1000000.0)
            self.root.child_Seq = ["" for _ in range(self.root.NMosse)]
            self.root.children = {}  # map of flattened moves to resulting MCTSNode
            self.root.RisultatoPerfetto = -1000
            self.root.SegnoGiocatore = +1
            if self.root.Board.ChiMuove == 2:
                self.root.SegnoGiocatore = -1

        self.NMossaMax = 0
    

    def ValoreToWinLose(self, Val, Komi):
        return math.tanh((Val - Komi) / 4)


    def PlayRootMove(self, Pos):

        for i in range(self.root.NMosse):
            if self.root.Mosse[i] == Pos:
                break

        if self.root.Mosse[i] != Pos:
            print("PlayRootMove ERRORE mossa non trovata!")
            return

        self.root = self.root.maybe_add_child(i)
        
        del self.root.parent.children



    def suggest_move(self, InitBoard:BOARD = None, initCPUct = 0.0, SceltaTemp = 0.0, ProbsTemp = 1.0, Komi = 0, VisVal = False):
        """Used for playing a single game.
        For parallel play, use initialize_move, select_leaf,
        incorporate_results, and pick_move
        """
        global CPUct
        CPUct = initCPUct

        if InitBoard != None:
            self.root.N = 0
            self.root.Board = InitBoard
            self.root.Mosse = self.root.Board.DammiMosse()
            self.root.NMosse = len(self.root.Mosse)
            self.root.is_expanded = False
            self.root.child_N = np.zeros([self.root.NMosse], dtype=np.float32)
            self.root.child_losses = np.zeros([self.root.NMosse], dtype=np.float32)
            self.root.child_W = np.zeros([self.root.NMosse], dtype=np.float32) 
            self.root.child_Wa = np.zeros([self.root.NMosse], dtype=np.float32) 
            self.root.child_prior = np.zeros([self.root.NMosse], dtype=np.float32)
            self.root.child_new = np.ones([self.root.NMosse], dtype=np.float32) * (1000000.0)
            self.root.child_Seq = ["" for _ in range(self.root.NMosse)]
            self.root.children = {}  # map of flattened moves to resulting MCTSNode
            self.root.SegnoGiocatore = +1
            if self.root.Board.ChiMuove == 2:
                self.root.SegnoGiocatore = -1
        
        self.root.Seq = ""
        self.Komi = Komi


        self.NMossaMax = 0

        if VisVal:
            TempoP = datetime.datetime.now()

        i = 0
        self.FoglieElaborate = 0
        FoglieElaborateOld = 0
        CicliSenzaFoglie = 0
        while self.FoglieElaborate < self.num_readouts and CicliSenzaFoglie < 10:
            i += 1
            FoglieElaborateOld = self.FoglieElaborate

            self.tree_search()

            if FoglieElaborateOld == self.FoglieElaborate:
                CicliSenzaFoglie += 1
            else:    
                CicliSenzaFoglie = 0


            if VisVal and (verbose >= 2 and (i % 9 == 0 or verbose >= 3)):

                if verbose >= 4:
                    self.root.VisDatiNodo()

                fcoord = self.root.best_child()
                Pos = self.root.Mosse[fcoord]
                Val = self.root.child_Q[fcoord]
                Alpha = self.root.child_A[fcoord]
                
                ProfMax = self.NMossaMax - self.root.Board.NMossa
                print("%d" % (self.FoglieElaborate), " Rn:%d" % self.root.N, " Mossa:", PosToString(Pos), "Val: %+1.3f" % Val, "(%+1.2f)" % Alpha, "ProfMax:", ProfMax, self.root.BestSeq, end='                 \r')
        
        if VisVal:
            print(end='\n')


        if VisVal:
            TempoA = datetime.datetime.now()
            TempoT = TempoA - TempoP            
            print("Sec:", TempoT.total_seconds(), "Mosse:", (self.FoglieElaborate), "M/s:", (self.FoglieElaborate)/TempoT.total_seconds())


        if VisVal and verbose >= 3:
            self.root.VisDatiNodo()

        fcoord = self.root.best_child()
        Pos = self.root.Mosse[fcoord]
        Val = self.root.child_Q[fcoord] 
        Alpha = self.root.child_A[fcoord]
        
        ProfMax = self.NMossaMax - self.root.Board.NMossa
        if VisVal:
            print("%d" % (self.FoglieElaborate), " Rn:%d" % self.root.N, "Mossa:", PosToString(Pos), "Val: %+1.3f" % Val, "(%+1.2f)" % Alpha, "ProfMax:", ProfMax)


        Best_child = self.root.best_child()
        BestPos = self.root.Mosse[Best_child]
        Val = self.root.child_Q[Best_child] 
        Alpha = self.root.child_A[Best_child]

        if SceltaTemp > 0.1:
            counts = [x ** (1. / SceltaTemp) for x in self.root.child_N]
            counts_sum = float(sum(counts))
            iprobs = [x / counts_sum for x in counts]
            iScelta = np.random.choice(self.root.NMosse, None, False, iprobs)
            
            Scelta = self.root.Mosse[iScelta]
        else:
            Scelta = BestPos


        if ProbsTemp < 0.1:
            ProbsTemp = 0.1

        if ProbsTemp >= 0.1:
            counts = [x ** (1. / ProbsTemp) for x in self.root.child_N]
            counts_sum = float(sum(counts))
            iprobs = [x / counts_sum for x in counts]

            Probs = [0] * (Board.Size * Board.Size)
            for i in range(len(iprobs)):
                Probs[ CoordBoardToZero(self.root.Mosse[i]) ] = iprobs[i]

        if VisVal:
            print("Scelta:", PosToString(Scelta), "Val: %+1.3f" % Val, "(%+1.2f)" % Alpha)

        return Scelta, BestPos, Val, Alpha, Probs



    def tree_search(self, parallel_readouts=None):
        if parallel_readouts is None:
            parallel_readouts = min(minparallel_readouts, self.num_readouts)

        leaves = []

        FoglieBatch = 0
        Cicli = 0
        while FoglieBatch < parallel_readouts and Cicli < parallel_readouts * 10:

            Cicli += 1
            FoglieBatch += 1
            self.FoglieElaborate += 1

            leaf = self.root.select_leaf()

            if leaf.Board.NMossa > self.NMossaMax:
                self.NMossaMax = leaf.Board.NMossa                

            # if game is over, override the value estimate with the true score
            if leaf.is_done():

                if leaf.RisultatoPerfetto != -1000:                
                    value_a = leaf.RisultatoPerfetto

                    # Se incontro una foglia già risolta non la considero nel calcolo delle foglie elaborate
                    if leaf != self.root:
                        FoglieBatch -= 1
                else:
                    if leaf.Board.NMossa >= Board.MossaFinalePerfetto and leaf.Board.ChiMuove != 0:
                        self.NMossaMax = Board.MossaFinale

                        Vuote = leaf.Board.DammiVuote()
                        self.Scacc.Table[leaf.Board.NMossa] = leaf.Board

                        value_a = leaf.SegnoGiocatore * -self.Eval.Valuta_Finale(self.Scacc, Vuote, leaf.Board.NMossa, (leaf.SegnoGiocatore * self.Komi)-16, (leaf.SegnoGiocatore * self.Komi)+16)
                    else:
                        value_a = leaf.Board.NPezzi[1] - leaf.Board.NPezzi[2]

                    leaf.RisultatoPerfetto = value_a


                value = self.ValoreToWinLose(value_a, self.Komi)

                if leaf != self.root:
                    leaf.backup_value(value, value_a, up_to=self.root)
                    continue


            leaf.add_virtual_loss(up_to=self.root)
            leaves.append(leaf)


        if leaves:
            move_probs, values, alpha, beta = self.NeuralNet.predici_board([leaf.Board.DammiBoardDaPredire() for leaf in leaves], [(leaf.SegnoGiocatore * self.Komi) for leaf in leaves])

            for leaf, move_prob, value, value_a in zip(leaves, move_probs, values, alpha):
                leaf.revert_virtual_loss(up_to=self.root)

                move_prob_i = np.asarray([move_prob[ CoordBoardToZero(Pos) ] for Pos in leaf.Mosse])

                leaf.incorporate_results(move_prob_i, (leaf.SegnoGiocatore * value), (leaf.SegnoGiocatore * value_a), up_to=self.root)

        return leaves


