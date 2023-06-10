from typing import Any, Tuple
from torchvision.datasets import MNIST
from torchvision import transforms as tf  
from torch import nn 


class MnistAutoencoderDataset(MNIST):
    def __init__(self, type_:str,  *args, **kwargs): 
        super().__init__(*args, **kwargs)
        self.type_ = type_
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, _ = super().__getitem__(index)
        if self.type_ == "fc":
            return nn.Flatten()(img), nn.Flatten()(img), _
        elif self.type_ == "conv": 
            return img, img, _
        elif self.type_ == "conv_crop":
            return tf.RandomErasing(p=1, 
                                    # ratio=(0.2, 0.2),
                                    value=0.0)(img), img, _
        else: 
            raise ValueError(f"type_ must be in [conv, fc, conv_crop]")