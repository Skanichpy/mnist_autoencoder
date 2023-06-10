import matplotlib.pyplot as plt 
import torch 
from torch import nn 

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def get_gt(loader):
    batch_gt = [] 
    for _, _, gt in loader: 
        batch_gt.append(gt)

    return torch.cat(batch_gt).view(-1)


def plot_before_after(autoenc, x:torch.Tensor, is_flatten=True):
    """x must be a squared image""" 
    fig, axis = plt.subplots(1,2,figsize=(12, 7))
    if is_flatten:
        flatten_dim = x.shape[1]
        new_d = int(flatten_dim**.5)
        after = autoenc(x).view(new_d, new_d, 1).detach().numpy()
        before = x.view(new_d, new_d, 1)
    else: 
        after = autoenc(x)[0].permute(1,2,0).detach().numpy()
        before = x[0].permute(1,2,0)
    axis[0].imshow(before, cmap='gray')
    axis[0].set_title("BEFORE")
    axis[1].imshow(after, cmap='gray')
    axis[1].set_title("AFTER")


def plot_hidden_distr(loader, trainer, autoencoder):
    truth = get_gt(loader)
    encoded = torch.cat(trainer.predict(autoencoder, loader))
    encoded = nn.Flatten()(encoded)

    encoded = encoded.detach().cpu().numpy()
    truth = truth.detach().cpu().numpy() 

    tsne = TSNE(n_components=2)

    for t in range(truth.min(), truth.max()+1):
        condition = truth == t
        X_emb = tsne.fit_transform(encoded[condition, :])
        x1, x2 = X_emb[:,0], X_emb[:, 1]
        plt.scatter(x1,
                    x2,
                    label=t)
        plt.legend();



    
