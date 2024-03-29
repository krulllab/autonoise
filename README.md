# Towards Structured Noise Models for Unsupervised Denoising

Benjamin Salmon<sup>1</sup> and Alexander Krull<sup>2</sup></br>
<sup>1,2</sup>School of Computer Science, University of Birmingham<br>
<sup>1</sup>brs209@student.bham.ac.uk<br>
<sup>2</sup>a.f.f.krull@bham.ac.uk <br>

![teaserFigure](resources/teaser.png)

The introduction of unsupervised methods in denoising has shown that unpaired noisy data can be used to train denoising networks, which can not only produce high quality results but also enable us to sample multiple possible diverse denoising solutions. 
However, these systems rely on a probabilistic description of the imaging noise--a noise model.
Until now, imaging noise has been modelled as pixel-independent in this context.
While such models often capture shot noise and readout noise very well, they are unable to describe many of the complex patterns that occur in real life applications.
Here, we introduce a novel learning-based autoregressive noise model to describe imaging noise and show how it can enable unsupervised denoising for settings with complex structured noise patterns.
We explore different ways to train a model for real life imaging noise and show that our deep autoregressive noise model has the potential to greatly improve denoising quality in structured noise datasets.
We showcase the capability of our approach on various simulated datasets and on real photo-acoustic imaging data.

### Information

Code for the publication [Towards Structured Noise Models for Unsupervised Denoising](https://link.springer.com/chapter/10.1007/978-3-031-25069-9_25). 

### BibTeX

```
@inproceedings{salmon2022towards,
  title={Towards Structured Noise Models for Unsupervised Denoising},
  author={Salmon, Benjamin and Krull, Alexander},
  booktitle={European Conference on Computer Vision},
  pages={379--394},
  year={2022},
  organization={Springer}
}
```

### Dependencies
We recommend installing the dependencies in a conda environment. If you haven't already, install miniconda on your system by following this [link](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html).<br>
Once conda is installed, create and activate an environment by entering these lines into a command line interface:<br>
1. `conda create --name autonoise`
2. `conda activate autonoise`


Next, install PyTorch and torchvision for your system by following this [link](https://pytorch.org/get-started/locally/).<br> 
After that, you're ready to install the dependencies for this repository:<br>
`pip install lightning jupyterlab matplotlib tifffile scikit-image tensorboard`

### Getting Started
Data used in the paper can be found at (https://zenodo.org/record/7010202#.Yv_Uyy8w1QI).
The 'examples' directory contains notebooks for denoising the Convallaria with simulated sCMOS noise dataset. Notebooks assume data has been stored as .tif files in a 'data' directory as numpy ndarrays with dimensions [Number, Channels, Height, Width].
