import torch
import numpy as np

class nm_dataset(torch.utils.data.Dataset):
    """Creates pytorch dataset for training the noise model.

    Parameters
    ----------
    n_data : numpy ndarray
        The noise image data.
    transform : torchvision.transforms function, optional
        Transformation to be applied to the images. The default is None.

    """
    def __init__(self, n_data, transform=None):
        self.n_data = torch.from_numpy(n_data).type(torch.float)
        
        self.transform = transform
    
        if self.n_data.dim() == 3:
            self.n_data = self.n_data[:, np.newaxis]
        elif self.n_data.dim() != 4:
            print('Data dimensions should be [B,C,H,W] or [B,H,W]')
                
    def getparams(self):
        noise_mean = torch.mean(self.n_data)
        noise_std = torch.std(self.n_data)
        return noise_mean, noise_std
    
    def __len__(self):
        return self.n_data.shape[0]
    
    def __getitem__(self, idx):
        n = self.n_data[idx]
        
        if self.transform:
            n = self.transform(n)
        
        return n
        
def create_nm_loader(n_data, split=0.8, batch_size=32, transform=None):
    """Creates pytorch dataloaders for training the noise model.
    
    Parameters
    ----------
    n_data : numpy ndarray
        The noise image data.
    split : Float, optional
        Percent of data to go into the training set, remaining
        data will go into the validation set. The default is 0.8.
    batch_size : int, optional
        Size of batches. The default is 32.
    transform : torchvision.transforms function, optional
        Transformation to be applied to the images. The default is None.

    Returns
    -------
    train_loader : torch.utils.data.DataLoader class
        Pytorch dataloader of the training set.
    val_loader : torch.utils.data.DataLoader class
        Pytorch dataloader of the validation set.
    noise_mean : float
        Mean of the noise data, used to normalise.
    noise_std : float
        Standard deviation of the noise data, used to normalise.

    """
    dataset = nm_dataset(n_data, transform)
    
    train_set, val_set = torch.utils.data.random_split(dataset, [round(len(dataset)*split), round(len(dataset)*(1-split))])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=True)

    noise_mean, noise_std = dataset.getparams()
    return train_loader, val_loader, noise_mean, noise_std
    

class dn_dataset(torch.utils.data.Dataset):
    """Creates pytorch dataset for training the denoising VAE.

    Parameters
    ----------
    x_data : numpy ndarray
        The noisy image data.
    transform : torchvision.transforms function, optional
        Transformation to be applied to the images. The default is None.

    """
    def __init__(self, x_data, transform=None):
        
        self.x_data = torch.from_numpy(x_data).type(torch.float)
        
        self.transform = transform
        
        if self.x_data.dim() == 3:
            self.x_data = self.x_data[:,np.newaxis,...]
        elif self.x_data.dim() != 4:
            print('Data dimensions should be [B,C,H,W] or [B,H,W]')
    
    def getimgshape(self):
        img = self.__getitem__(0)
        return (img.shape[1], img.shape[2])
    
    def getparams(self):
        return torch.mean(self.x_data), torch.std(self.x_data)
    
    def __len__(self):
        return self.x_data.shape[0]
    
    def __getitem__(self, idx):
        x = self.x_data[idx]
        
        if self.transform:
            x = self.transform(x)
                
        return x

def create_dn_loader(x_data, split=0.8, batch_size=32, transform=None):
    """Creates pytorch dataloaders for training the denoising VAE.
    
    Parameters
    ----------
    x_data : numpy ndarray
        The noisy image data.
    split : Float, optional
        Percent of data to go into the training set, remaining
        data will go into the validation set. The default is 0.8.
    batch_size : int, optional
        Size of batches. The default is 32.
    transform : torchvision.transforms function, optional
        Transformation to be applied to the images. The default is None.

    Returns
    -------
    train_loader : torch.utils.data.DataLoader class
        Pytorch dataloader of the training set.
    val_loader : torch.utils.data.DataLoader class
        Pytorch dataloader of the validation set.
    img_shape : list of ints
        Height and width of the images, used to prepare
        the VAE.
    data_mean : float
        Mean of the noisy image data, used to normalise.
    data_std : float
        Standard deviation of the noisy image data, used to normalise.

    """
    dataset = dn_dataset(x_data, transform)
    
    img_shape = dataset.getimgshape()
    
    data_mean, data_std = dataset.getparams()
    
    train_set, val_set = torch.utils.data.random_split(dataset, [round(len(dataset)*split), round(len(dataset)*(1-split))])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=True)
        
    return train_loader, val_loader, img_shape, data_mean, data_std
