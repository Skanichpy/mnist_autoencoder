from torchvision import transforms as tf 
from torch import nn 
import torch 

CONFIG = dict(
        MNIST_DIR = "data",
        TRANSFORMS = tf.Compose([tf.ToTensor(),
                                 tf.Normalize(mean=0.5, std=0.5)]
                                ),

        
        TRAIN_BATCH_SIZE = 200 ,
        TEST_BATCH_SIZE = 200,
        ACCELERATOR= "gpu",

        hyperparams =  dict(
            fc = dict(input_size = 784,
                      context_size = 2,
                      num_layers_enc = 6,
                      num_layers_dec = 8,
                      activation = nn.ReLU(),
                      loss = torch.nn.functional.mse_loss,
                      lr = 0.001
                      ),

            conv = dict(IN_CHANNELS=1)
        
        )
    )

