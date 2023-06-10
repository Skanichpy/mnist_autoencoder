from model_package.abstract_autoencoder import AbstractAutoEncoder
from torch import nn 
import torch

class FcAutoEncoder(AbstractAutoEncoder): 

    def __init__(self, input_size:int, context_size: int,
                 num_layers_enc:int, num_layers_dec:int, 
                 activation: callable, 
                 loss:int, lr:float) -> None: 
        super().__init__(loss, lr)

        self.encoder = self.__build_encoder(input_size, context_size, activation,
                             num_layers_enc) 
        self.decoder = self.__build_decoder(input_size, context_size, activation,
                             num_layers_dec) 

    def __build_encoder(self, input_size, context_size, 
                        activation, num_layers):
        
        hiddens = self.__generate_hiddens(input_size, context_size,
                                          num_layers)
        encoder = self.__generate_sequential(hiddens, activation)
        return encoder 

    def __build_decoder(self, input_size, context_size, 
                        activation, num_layers):
        hiddens = self.__generate_hiddens(context_size, input_size,
                                          num_layers)
        decoder = self.__generate_sequential(hiddens, activation)
        return decoder

    def __generate_hiddens(self, from_size, to_size, num_layers): 
        hiddens = torch.linspace(from_size, to_size, num_layers).long().view(1,-1)
        hiddens = torch.cat([hiddens[:, :-1], hiddens[:, 1:]], dim=0).transpose(1,0)
        return hiddens
    
    def __generate_sequential(self, hiddens: torch.Tensor, 
                              activation:callable): 
        seq = nn.Sequential()
        for idx, hidden in enumerate(hiddens): 
            seq.append(nn.Linear(*hidden))
            if idx != (hiddens.shape[0] - 1):
                seq.append(activation)
        return seq 

    

class ConvAutoEncoder(AbstractAutoEncoder): 
    def __init__(self, in_channels, loss, lr, context_size): 
        super().__init__(loss, lr)
        self.in_channels = in_channels
        self.context_size = context_size

        self.encoder = self.__build_encoder(in_channels)
        self.decoder = self.__build_decoder(in_channels)

    def __build_encoder(self, in_channels):
    
        return nn.Sequential(
                        nn.Conv2d(1, 8, 3, stride=2, padding=1),
                        nn.ReLU(True),
                        nn.Conv2d(8, 16, 3, stride=2, padding=1),
                        nn.BatchNorm2d(16),
                        nn.ReLU(True),
                        nn.Conv2d(16, 32, 3, stride=2, padding=0),
                        nn.ReLU(True),
                        nn.Flatten(),
                        nn.Linear(3*3*32,128),
                        nn.ReLU(),
                        nn.Linear(128, self.context_size)
                        )  
    
    def __build_decoder(self, out_channels):
    
        return nn.Sequential( 
                        nn.Linear(self.context_size, 128),
                        nn.ReLU(True),
                        nn.Linear(128, 3 * 3 * 32),
                        nn.ReLU(True),
                        nn.Unflatten(dim=1, 
                                     unflattened_size=(32, 3, 3)),
                        nn.ConvTranspose2d(32, 16, 3, 
                                           stride=2, output_padding=0),
                        nn.BatchNorm2d(16),
                        nn.ReLU(True),
                        nn.ConvTranspose2d(16, 8, 3, stride=2, 
                                           padding=1, output_padding=1),
                        nn.BatchNorm2d(8),
                        nn.ReLU(True),
                        nn.ConvTranspose2d(8, 1, 3, stride=2, 
                                    padding=1, output_padding=1),
                        nn.Tanh()
                        )
    
    def predict_step(self, batch, batch_idx): 
        x, _, _ = batch 
        return self.encoder(x)
    
