# MonoDepth-SD-ConvTrans-PyTorch
A semantic diffusion guided convolutional transformer network with the input of the same scene dual-resolution images is proposed.
We tested the performance of our model on the NYU Depth V2 Dataset (Official Split).

## Requirements

* Python 3
* pycharm
* PyTorch 
  * Tested with PyTorch 2.0.0
* CUDA 12.2 (if using CUDA)


## To Run

change the root path '/datasets/nyud_v2/' to your own downloaded nyudv2 offical split
The model i2d.pth was trained with nyudv2 raw dataset with 90k training data
```
testmodel_ada.py
```

## Data Processing
### NYU Depth V2 Dataset
* The [NYU Depth V2 dataset] contains a variety of indoor scenes, with 249 scenes for training and 215 scenes for testing. We used the official split for training and testing. This [github page] provides a very detailed walkthrough and matlab code for data processing.
* Following previous works, we used the official toolbox which uses the Colorization method proposed by Levin et al. to fill in the missing values of depth map in the training set.
* To make comparison with previous works, we evaluated our model using the official evaluation set of 654 densely labeled image pairs.

## Model

This paper presents a multi-resolution monocular depth estimation method based on smooth foundation depth and residual detail depth resolution. 
We call it the semantic diffusion guided convolutional transformer model (SD-ConvTrans). 
It uses the layered feature field of the convolutional model and the global context information strengthened by the Transformers model level to design a double-branch regression network. 
The network can maintain global consistency and obtain smooth base depth. The residual detail depth can also be contrasted to highlight the local context. 
In this paper, the semantically guided real boundary diffusion module is used to estimate the depth value at the boundary as an anisotropic diffusion process guided by globally consistent semantic features. 
Experiments on indoor and outdoor open source datasets show that the problems identified and the method proposed in this paper can greatly promote the monocular depth estimation.

## Loss Function

We employed three parts in the loss function in our model. The loss is a weighted sum of 3 parts: the depth loss, the gradient loss and the surface normal loss.

The weight ratio between the three loss was set to 1:1:1.

## References
* Eigen, D., Puhrsch, C., Fergus, R.: Depth map prediction from a single image using a multiscale
deep network. In: Advances in neural information processing systems (NIPS). (2014)
2366–2374
* Fu, Huan, Mingming Gong, Chaohui Wang and Dacheng Tao. “A Compromise Principle in Deep Monocular Depth Estimation.” CoRR abs/1708.08267 (2017): n. pag.
* R. Garg, V. Kumar, G. Carneiro, and I. Reid. Unsupervised cnn for single view depth estimation: Geometry to the rescue. In Proc. of the European Conf. on Computer Vision (ECCV), 2016.
* Geiger, Andreas, et al. "Vision meets robotics: The KITTI dataset." The International Journal of Robotics Research 32.11 (2013): 1231-1237.
* C. Godard, O. Mac Aodha, and G. J. Brostow. Unsupervised monocular depth estimation with left-right consistency. arXiv:1609.03677v2, 2016.
* Hu, Junjie, et al. "Revisiting Single Image Depth Estimation: Toward Higher Resolution Maps with Accurate Object Boundaries." arXiv preprint arXiv:1803.08673 (2018).
* Kuznietsov, Yevhen, Jörg Stückler, and Bastian Leibe. "Semi-supervised deep learning for monocular depth map prediction." Proc. of the IEEE Conference on Computer Vision and Pattern Recognition. 2017.
* I. Laina, C. Rupprecht, V. Belagiannis, F. Tombari, and N. Navab. Deeper depth prediction with fully convolutional residual networks. In Proc. of the Int. Conf. on 3D Vision (3DV), 2016.
* Levin, Anat, Dani Lischinski, and Yair Weiss. "Colorization using optimization." ACM Transactions on Graphics (ToG). Vol. 23. No. 3. ACM, 2004.
* Silberman, Nathan, et al. "Indoor segmentation and support inference from rgbd images." European Conference on Computer Vision. Springer, Berlin, Heidelberg, 2012.
