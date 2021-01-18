
import numpy as np
import os.path
from tqdm import tqdm
from prettytable import PrettyTable



import SaiOthelloNN 


InfoShowed = False

if SaiOthelloNN.use_pytorch == 1:
    import torch as pt


    class AverageMeter(object):
        """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

        def __init__(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def __repr__(self):
            return f'{self.avg:1.4f}'

        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


    class NETPytorch(pt.nn.Module):

        def res_net_block(self, input_data, filters, conv_size, i):

            x = self.convres1[i](input_data)
            x = self.batchnormres1[i](x)
            x = self.relures1[i](x)
            x = self.convres2[i](x)
            x = self.batchnormres2[i](x)
            x += input_data
            x = self.relures2[i](x)
            return x


        def out_blockAlpha(self, input_data, filters):

            x = self.convoutAlpha(input_data)
            x = self.batchnormoutAlpha(x)
            x = self.reluout1Alpha(x)
            x = x.view(-1, self.Size*self.Size*2)
            x = self.linear1outAlpha(x)
            x = self.reluout2Alpha(x)
            x = self.linear2outAlpha(x)
            return x

        def out_blockBeta(self, input_data, filters):

            x = self.convoutBeta(input_data)
            x = self.batchnormoutBeta(x)
            x = self.reluout1Beta(x)
            x = x.view(-1, self.Size*self.Size*2)
            x = self.linear1outBeta(x)
            x = self.reluout2Beta(x)
            x = self.linear2outBeta(x)
            x = self.reluout3Beta(x)
            return x


        def __init__(self, Size, l2a=0, l2k=0, Filters=144, NetDepth=12, usecuda=False):

            self.Size = Size
            self.Filters = Filters
            self.NetDepth = NetDepth
            self.l2a = l2a
            self.l2k = l2k
            self.usecuda = usecuda

            super(NETPytorch, self).__init__()
       
            self.conv1 = pt.nn.Conv2d(1, Filters, kernel_size=3, padding = (1, 1), bias = False)
            self.batchnorm1 = pt.nn.InstanceNorm2d(Filters)
            self.relu1 = pt.nn.ReLU()

            self.convres1 = []
            self.batchnormres1 = []
            self.relures1 = []
            self.convres2 = []
            self.batchnormres2 = []
            self.relures2 = []

            for i in range(NetDepth):
                self.convres1.append(pt.nn.Conv2d(Filters, Filters, kernel_size=3, stride = 1, padding = (1, 1), bias = False))
                self.batchnormres1.append(pt.nn.InstanceNorm2d(Filters))
                self.relures1.append(pt.nn.ReLU())
                self.convres2.append(pt.nn.Conv2d(Filters, Filters, kernel_size=3, stride = 1, padding = (1, 1), bias = False))
                self.batchnormres2.append(pt.nn.InstanceNorm2d(Filters))
                self.relures2.append(pt.nn.ReLU())

            self.convres1 = pt.nn.ModuleList(self.convres1)
            self.batchnormres1 = pt.nn.ModuleList(self.batchnormres1)
            self.relures1 = pt.nn.ModuleList(self.relures1)
            self.convres2 = pt.nn.ModuleList(self.convres2)
            self.batchnormres2 = pt.nn.ModuleList(self.batchnormres2)
            self.relures2 = pt.nn.ModuleList(self.relures2)


            self.convoutAlpha = pt.nn.Conv2d(Filters, 2, kernel_size=1, bias = False)
            self.batchnormoutAlpha = pt.nn.InstanceNorm2d(2)
            self.reluout1Alpha = pt.nn.ReLU()
            self.linear1outAlpha = pt.nn.Linear(self.Size*self.Size*2, self.Size*self.Size*2)
            self.reluout2Alpha = pt.nn.ReLU()
            self.linear2outAlpha = pt.nn.Linear(self.Size*self.Size*2, 1)

            self.convoutBeta = pt.nn.Conv2d(Filters, 2, kernel_size=1, bias = False)
            self.batchnormoutBeta = pt.nn.InstanceNorm2d(2)
            self.reluout1Beta = pt.nn.ReLU()
            self.linear1outBeta = pt.nn.Linear(self.Size*self.Size*2, self.Size*self.Size*2)
            self.reluout2Beta = pt.nn.ReLU()
            self.linear2outBeta = pt.nn.Linear(self.Size*self.Size*2, 1)
            self.reluout3Beta = pt.nn.ELU()

            self.convout2 = pt.nn.Conv2d(Filters, 2, kernel_size=1, bias = False)
            self.relu2 = pt.nn.ReLU()
            self.batchnorm2 = pt.nn.InstanceNorm2d(2)
            self.linear2 = pt.nn.Linear(self.Size*self.Size*2, self.Size * self.Size)
            self.softmax2 = pt.nn.Softmax(dim=1)

            self.tanh = pt.nn.Tanh()


        def forward(self, input_data, Komi):
        
            input_data = pt.from_numpy(np.asarray(input_data)).float()
            Komi = pt.from_numpy(np.asarray(Komi)).float()
            if self.usecuda:
                input_data = input_data.contiguous().cuda()
                Komi = Komi.contiguous().cuda()

            input_data = input_data.view(-1, 1, self.Size, self.Size)

            x = self.conv1(input_data)
            x = self.batchnorm1(x)
            x = self.relu1(x)

            # blocco deep residual comune
            r = self.res_net_block(x, self.Filters, 3, 0)
            for i in range(self.NetDepth - 1):
                r = self.res_net_block(r, self.Filters, 3, i+1)
            
            outputs_m = self.convout2(r)
            outputs_m = self.relu2(outputs_m)
            outputs_m = self.batchnorm2(outputs_m)
            outputs_m = outputs_m.view(-1, self.Size*self.Size*2)
            outputs_m = self.linear2(outputs_m)
            outputs_m = self.softmax2(outputs_m)
            
            outputs_Alpha = self.out_blockAlpha(r, self.Filters)
            outputs_Beta = self.out_blockBeta(r, self.Filters)

            Sigma = self.tanh(outputs_Beta * (outputs_Alpha - Komi))

            return outputs_m, Sigma, outputs_Alpha, outputs_Beta


    class NEURALNETPytorch():

        def __init__(self, Size, net_file=None, learn_rate=0.001, l2a=0, l2k=0, Filters=144, NetDepth=12):

            self.Size = Size
            self.net_file = net_file
            self.Filters = Filters
            self.NetDepth = NetDepth
            self.LearnRate = learn_rate
            
            self.cuda = False
            if pt.cuda.is_available():
                self.cuda = True

            self.NNet = NETPytorch(Size, l2a=0, l2k=0, Filters=Filters, NetDepth=NetDepth, usecuda=self.cuda)

            if pt.cuda.is_available():
                self.NNet.cuda()
            
           
            # Se ho specificato il nome del file di una Rete da caricare allora la
            # carico
            if net_file != None and os.path.exists(net_file):
                self.NNet.load_state_dict(pt.load(net_file))


            table = PrettyTable(["Modules", "Parameters"])
            total_params = 0
            for name, parameter in self.NNet.named_parameters():
                if not parameter.requires_grad: 
                    continue
                param = parameter.numel()
                table.add_row([name, param])
                total_params+=param
            print(table)
            print(f"Total Trainable Params: {total_params}")


        def evaluate(self, board_batch, Mosse_batch, pedine_batch, komi_batch, batch_size=256):
            """
            examples: list of examples, each example is of form (board, pi, v)
            """
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            examples = []

            for i in range(len(board_batch)):
                examples.append([board_batch[i], Mosse_batch[i], pedine_batch[i], komi_batch[i]])

            batch_count = int(len(examples) / batch_size)

            self.NNet.eval()

            t = tqdm(range(batch_count), desc='Evaluate Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=batch_size)

                boards, pis, vs, komi = list(zip(*[examples[i] for i in sample_ids]))
                boards = pt.FloatTensor(np.array(boards).astype(np.float64))
                komi = pt.FloatTensor(np.array(komi).astype(np.float64))
                target_pis = pt.FloatTensor(np.array(pis))
                target_vs = pt.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if self.cuda:
                    target_pis, target_vs = target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                # compute output
                with pt.no_grad():
                    out_pi, out_v, out_a, out_b = self.NNet(boards, komi)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi/60 + l_v

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

            return [v_losses.avg, pi_losses.avg]
            

        def train(self, board_batch, Mosse_batch, pedine_batch, komi_batch, batch_size=256, epochs=10):
            """
            examples: list of examples, each example is of form (board, pi, v)
            """
            optimizer = pt.optim.Adam(self.NNet.parameters(), lr=self.LearnRate)

            examples = []
            for i in range(len(board_batch)):
                examples.append([board_batch[i], Mosse_batch[i], pedine_batch[i], komi_batch[i]])

            for epoch in range(epochs):
                print('EPOCH ::: ' + str(epoch + 1))
                self.NNet.train()
                pi_losses = AverageMeter()
                v_losses = AverageMeter()

                batch_count = int(len(examples) / batch_size)

                t = tqdm(range(batch_count), desc='Training Net')
                for _ in t:
                    sample_ids = np.random.randint(len(examples), size=batch_size)
                    
                    boards, pis, vs, komi = list(zip(*[examples[i] for i in sample_ids]))
                    boards = pt.FloatTensor(np.array(boards).astype(np.float64))
                    komi = pt.FloatTensor(np.array(komi).astype(np.float64))
                    target_pis = pt.FloatTensor(np.array(pis))
                    target_vs = pt.FloatTensor(np.array(vs).astype(np.float64))

                    if self.cuda:
                        target_pis, target_vs = target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                    # compute output
                    out_pi, out_v, out_a, out_b = self.NNet(boards, komi)
                    l_pi = self.loss_pi(target_pis, out_pi)
                    l_v = self.loss_v(target_vs, out_v)
                    total_loss = l_pi/60 + l_v

                    # record loss
                    pi_losses.update(l_pi.item(), boards.size(0))
                    v_losses.update(l_v.item(), boards.size(0))
                    t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                    # compute gradient and do SGD step
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                


        def loss_pi(self, targets, outputs):
            return pt.nn.BCELoss(reduction='sum')(outputs, targets) 

        def loss_v(self, targets, outputs):
            return pt.nn.MSELoss()(targets, outputs)




        def Salva(self):
            pt.save(self.NNet.state_dict(), self.net_file)



            # Predice le mosse ed il valore data una board
        def predici_board(self, input_board, Komi):

            self.NNet.eval()
            with pt.no_grad():
                pi, v, Alpha, Beta = self.NNet(input_board, Komi)

            return pi.cpu(), v.cpu()[0], Alpha.cpu(), Beta.cpu()






else:
    import tensorflow as tf
    import tensorflow.keras as keras
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    from tensorflow.keras import layers, models

    class NEURALNET:


        def res_net_block(self, input_data, filters, conv_size, l2a, l2k):
            x = layers.Conv2D(filters, conv_size, activation='relu', padding='same', use_bias = False)(input_data)
            x = layers.BatchNormalization(axis=3)(x)
            x = layers.Conv2D(filters, conv_size, activation=None, padding='same', use_bias = False)(x)
            x = layers.BatchNormalization(axis=3)(x)
            x = layers.Add()([x, input_data])
            x = layers.Activation('relu')(x)
            return x

        def out_blockAlpha(self, input_data, l2a, l2k):
            x = layers.Conv2D(2, 1, activation='relu', padding='same', use_bias = False)(input_data)
            x = layers.BatchNormalization(axis=3)(x)
            x = layers.Flatten()(x)
            x = layers.Dense(self.Size*self.Size*2, activation='relu')(x)
            x = layers.Dense(1, activation='linear', name='A')(x)
            return x

        def out_blockBeta(self, input_data, l2a, l2k):
            x = layers.Conv2D(2, 1, activation='relu', padding='same', use_bias = False)(input_data)
            x = layers.BatchNormalization(axis=3)(x)
            x = layers.Flatten()(x)
            x = layers.Dense(self.Size*self.Size*2, activation='relu')(x)
            x = layers.Dense(1, activation='exponential', name='B')(x)
            return x


        def __init__(self, Size, net_file=None, learn_rate=0.001, l2a=0, l2k=0, Filters=144, NetDepth=12):

            self.Size = Size
            self.net_file = net_file


            #policy = mixed_precision.Policy('mixed_float16')
            #mixed_precision.set_policy(policy)


           
            # Se ho specificato il nome del file di una Rete da caricare allora la
            # carico
            if net_file != None and os.path.exists(net_file):
                self.model = keras.models.load_model(net_file, compile = False)
            else:

                filters = Filters
                depth_r = NetDepth  # profondità residual tower comune
                depth_m = 0         # profondità residual tower per le mosse
                depth_v = 0         # profondità residual tower per il valore
                                
                
                inputs = keras.Input(shape=(self.Size, self.Size, 1))
                input_Komi = keras.Input(shape=(1,))

                x = layers.Conv2D(filters, 3, activation='relu', padding='same', use_bias = False)(inputs)
                x = layers.BatchNormalization(axis=1)(x)
                
                # blocco deep residual comune
                r = self.res_net_block(x, filters, 3, l2a, l2k)
                for _ in range(depth_r - 1):
                    r = self.res_net_block(r, filters, 3, l2a, l2k)
                
                if depth_m == 0:
                    rm = r
                else:
                    # blocco deep residual per le mosse
                    rm = self.res_net_block(r, filters, 3, l2a, l2k)
                    for _ in range(depth_m - 1):
                        rm = self.res_net_block(rm, filters, 3, l2a, l2k)
                
                if depth_v == 0:
                    rv = r
                else:
                    # blocco deep residual per il valore
                    rv = self.res_net_block(r, filters, 3, l2a, l2k)
                    for _ in range(depth_v - 1):
                        rv = self.res_net_block(rv, filters, 3, l2a, l2k)
                
                outputs_m = layers.Conv2D(2, 1, activation='relu', padding='same', use_bias = False)(rm)
                outputs_m = layers.BatchNormalization(axis=3)(outputs_m)
                outputs_m = layers.Flatten()(outputs_m)   
                outputs_m = layers.Dense(self.Size * self.Size)(outputs_m)
                outputs_m = layers.Activation('softmax', name='Mp', dtype='float32')(outputs_m)
                
                output_Alpha = self.out_blockAlpha(rv, l2a, l2k)
                output_Beta = self.out_blockBeta(rv, l2a, l2k)

                Komi = layers.Lambda(lambda x: x, name='Komi', dtype='float32')(input_Komi)
                Alpha_Komi = layers.Add(dtype='float32')([output_Alpha, -Komi])

                preSigma = keras.activations.tanh((layers.Multiply(dtype='float32')([output_Beta, Alpha_Komi])))
                Sigma = keras.layers.Lambda(lambda x: x, name='Sigma', dtype='float32')(preSigma)

                self.model = keras.Model(inputs=[inputs, input_Komi], outputs=[outputs_m, Sigma, output_Alpha, output_Beta])


            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = learn_rate), loss=[keras.losses.CategoricalCrossentropy(), keras.losses.MeanSquaredError(), None, None], metrics=["accuracy"], run_eagerly=False)

            global InfoShowed
            if not InfoShowed:
                self.model.summary()
                InfoShowed = True
        

        def Salva(self, NetPath = None):
            if NetPath == None:
                keras.models.save_model(self.model, self.net_file)
            else:
                keras.models.save_model(self.model, NetPath)

              


        # Predice le mosse ed il valore data una board
        def predici_board(self, input_board, komi):
            return self.model.predict_on_batch([np.reshape(input_board, [-1, self.Size, self.Size]), np.reshape(komi, [-1, 1])])







