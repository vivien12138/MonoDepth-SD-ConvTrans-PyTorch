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
[1] R. Ranftl, K. Lasinger, D. Hafner, K. Schindler, and V. Koltun, “Towards robust monocular depth estimation: Mixing datasets for zero-shot crossdataset transfer,” IEEE transactions on pattern analysis and machine intelligence, vol. 44, no. 3, pp. 1623–1637, 2020.
[2] D. Scharstein and R. Szeliski, “A taxonomy and evaluation of dense two-frame stereo correspondence algorithms,” International journal of computer vision, vol. 47, pp. 7–42, 2002.
[3] W. Han, J. Yin, X. Jin, X. Dai, and J. Shen, “Brnet: Exploring comprehensive features for monocular depth estimation,” in European Conference on Computer Vision. Springer, 2022, pp. 586–602.
[4] S. S. Tomar, M. Suin, and A. Rajagopalan, “Hybrid transformer based feature fusion for self-supervised monocular depth estimation,” in European Conference on Computer Vision. Springer, 2022, pp. 308–326.
[5] D. Eigen, C. Puhrsch, and R. Fergus, “Depth map prediction from a single image using a multi-scale deep network,” Advances in neural information processing systems, vol. 27, 2014.
[6] I. Laina, C. Rupprecht, V. Belagiannis, F. Tombari, and N. Navab,“Deeper depth prediction with fully convolutional residual networks,” in 2016 Fourth international conference on 3D vision (3DV). IEEE, 2016, pp. 239–248.
[7] Y. Cao, Z. Wu, and C. Shen, “Estimating depth from monocular images as classification using deep fully convolutional residual networks,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 28, no. 11, pp. 3174–3182, 2017.
[8] H. Fu, M. Gong, C. Wang, K. Batmanghelich, and D. Tao, “Deep ordinal regression network for monocular depth estimation,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 2002–2011.
[9] T. Chen, S. An, Y. Zhang, C. Ma, H. Wang, X. Guo, and W. Zheng, “Improving monocular depth estimation by leveraging structural awareness and complementary datasets,” in Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XIV 16. Springer, 2020, pp. 90–108.
[10] S. F. Bhat, I. Alhashim, and P. Wonka, “Adabins: Depth estimation using adaptive bins,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 4009–4018.
[11] X. Xu, J. Qiu, X. Wang, and Z. Wang, “Relationship spatialization for depth estimation,” in European Conference on Computer Vision. Springer, 2022, pp. 615–637.
[12] V. Hedau, D. Hoiem, and D. Forsyth, “Thinking inside the box: Using appearance models and context based on room geometry,” in Computer Vision–ECCV 2010: 11th European Conference on Computer Vision, Heraklion, Crete, Greece, September 5-11, 2010, Proceedings, Part VI 11. Springer, 2010, pp. 224–237.
[13] B.-s. Kim, P. Kohli, and S. Savarese, “3d scene understanding by voxelcrf,” in Proceedings of the IEEE International Conference on Computer Vision, 2013, pp. 1425–1432.
[14] A. Saxena, M. Sun, and A. Y. Ng, “Make3d: Learning 3d scene structure from a single still image,” IEEE transactions on pattern analysis and machine intelligence, vol. 31, no. 5, pp. 824–840, 2008.
[15] C. Hane, L. Ladicky, and M. Pollefeys, “Direction matters: Depth estimation with a surface normal classifier,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2015, pp. 381– 389.
[16] R. Furukawa, R. Sagawa, and H. Kawasaki, “Depth estimation using structured light flow–analysis of projected pattern flow on an object’s surface,” in Proceedings of the IEEE international conference on computer vision, 2017, pp. 4640–4648.
[17] W. Zhuo, M. Salzmann, X. He, and M. Liu, “Indoor scene structure analysis for single image depth estimation,” in Proceedings of the IEEE Conference on Computer Vision and Pattern recognition, 2015, pp. 614– 622.
[18] K. Karsch, C. Liu, and S. B. Kang, “Depth transfer: Depth extraction from video using non-parametric sampling,” IEEE transactions on pattern analysis and machine intelligence, vol. 36, no. 11, pp. 2144–2158, 2014.
[19] D. Eigen and R. Fergus, “Predicting depth, surface normals and semantic labels with a common multi-scale convolutional architecture,” in Proceedings of the IEEE international conference on computer vision, 2015, pp. 2650–2658.
[20] I. Alhashim and P. Wonka, “High quality monocular depth estimation via transfer learning,” arXiv preprint arXiv:1812.11941, 2018.
[21] J. Hu, M. Ozay, Y. Zhang, and T. Okatani, “Revisiting single image depth estimation: Toward higher resolution maps with accurate object boundaries,” in 2019 IEEE winter conference on applications of computer vision (WACV). IEEE, 2019, pp. 1043–1051.
[22] L. Wang, J. Zhang, Y. Wang, H. Lu, and X. Ruan, “Cliffnet for monocular depth estimation with hierarchical embedding loss,” in European Conference on Computer Vision. Springer, 2020, pp. 316–331.
[23] J.-H. Lee and C.-S. Kim, “Multi-loss rebalancing algorithm for monocular depth estimation,” in Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XVII 16. Springer, 2020, pp. 785–801.
[24] S. Li, J. Shi, W. Song, A. Hao, and H. Qin, “Hierarchical object relationship constrained monocular depth estimation.” Pattern Recognition, vol. 120, p. 108116, 2021.
[25] C.-Y. Wu, J. Wang, M. Hall, U. Neumann, and S. Su, “Toward practical monocular indoor depth estimation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 3814–3824.
[26] L. Piccinelli, C. Sakaridis, and F. Yu, “idisc: Internal discretization for monocular depth estimation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 21 477–21 487.
[27] X. Yang, Z. Ma, Z. Ji, and Z. Ren, “Gedepth: Ground embedding for monocular depth estimation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 12 719–12 727.
[28] V. Guizilini, I. Vasiljevic, D. Chen, R. Ambrus, , and A. Gaidon, “Towards zero-shot scale-aware monocular depth estimation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 9233–9243.
[29] P. Wang, X. Shen, Z. Lin, S. Cohen, B. Price, and A. L. Yuille, “Towards unified depth and semantic prediction from a single image,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2015, pp. 2800–2809.
[30] F. Liu, C. Shen, G. Lin, and I. Reid, “Learning depth from single monocular images using deep convolutional neural fields,” IEEE transactions on pattern analysis and machine intelligence, vol. 38, no. 10, pp. 2024–2039, 2015.
[31] D. Xu, E. Ricci, W. Ouyang, X. Wang, and N. Sebe, “Multi-scale continuous crfs as sequential deep networks for monocular depth estimation,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 5354–5362.
[32] D. Xu, W. Wang, H. Tang, H. Liu, N. Sebe, and E. Ricci, “Structured attention guided convolutional neural fields for monocular depth estimation,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 3917–3925.
[33] Z. Xia, P. Sullivan, and A. Chakrabarti, “Generating and exploiting probabilistic monocular depth estimates,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 65–74.
[34] W. Yuan, X. Gu, Z. Dai, S. Zhu, and P. Tan, “New crfs: Neural window fully-connected crfs for monocular depth estimation. arxiv 2022,” arXiv preprint arXiv:2203.01502, 2022.
[35] S. F. Bhat, I. Alhashim, and P. Wonka, “Localbins: Improving depth estimation by learning local distributions,” in European Conference on Computer Vision. Springer, 2022, pp. 480–496.
[36] Z. Zhang, C. Xu, J. Yang, J. Gao, and Z. Cui, “Progressive hard-mining network for monocular depth estimation,” IEEE Transactions on Image Processing, vol. 27, no. 8, pp. 3691–3702, 2018.
[37] J.-H. Lee and C.-S. Kim, “Monocular depth estimation using relative depth maps,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 9729–9738.
[38] L. Wang, J. Zhang, O. Wang, Z. Lin, and H. Lu, “Sdc-depth: Semantic divide-and-conquer network for monocular depth estimation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 541–550.
[39] U. Ali, B. Bayramli, T. Alsarhan, and H. Lu, “A lightweight network for monocular depth estimation with decoupled body and edge supervision,” Image and Vision Computing, vol. 113, p. 104261, 2021.
[40] S. M. H. Miangoleh, S. Dille, L. Mai, S. Paris, and Y. Aksoy, “Boosting monocular depth estimation models to high-resolution via contentadaptive multi-resolution merging,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 9685–9694.
[41] B. Li, C. Shen, Y. Dai, A. Van Den Hengel, and M. He, “Depth and surface normal estimation from monocular images using regression on deep features and hierarchical crfs,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2015, pp. 1119–1127.
[42] X. Chen, X. Chen, and Z.-J. Zha, “Structure-aware residual pyramid network for monocular depth estimation,” arXiv preprint arXiv:1907.06023, 2019.
[43] M. Song, S. Lim, and W. Kim, “Monocular depth estimation using laplacian pyramid-based depth residuals,” IEEE transactions on circuits and systems for video technology, vol. 31, no. 11, pp. 4381–4393, 2021.
[44] C. Zhang, W. Yin, B. Wang, G. Yu, B. Fu, and C. Shen, “Hierarchical normalization for robust monocular depth estimation,” Advances in Neural Information Processing Systems, vol. 35, pp. 14 128–14 139, 2022.
[45] A. Zhang, Y. Ma, J. Liu, and J. Sun, “Promoting monocular depth estimation by multi-scale residual laplacian pyramid fusion,” IEEE Signal Processing Letters, vol. 30, pp. 205–209, 2023.
[46] M. Ramamonjisoa, Y. Du, and V. Lepetit, “Predicting sharp and accurate occlusion boundaries in monocular depth estimation using displacement fields,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 14 648–14 657.
[47] S. Lee, J. Lee, B. Kim, E. Yi, and J. Kim, “Patch-wise attention network for monocular depth estimation,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 35, no. 3, 2021, pp. 1873–1881.
[48] H. Xu and F. Li, “A multiscale dilated convolution and mixed-order attention-based deep neural network for monocular depth prediction,” SN Applied Sciences, vol. 5, no. 1, p. 24, 2023.
[49] A. Jan and S. Seo, “Monocular depth estimation using res-unet with an attention model,” Applied Sciences, vol. 13, no. 10, p. 6319, 2023.
[50] R. Ranftl, A. Bochkovskiy, and V. Koltun, “Vision transformers for dense prediction,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 12 179–12 188.
[51] E. Xie, W. Wang, Z. Yu, A. Anandkumar, J. M. Alvarez, and P. Luo,“Segformer: Simple and efficient design for semantic segmentation with transformers,” Advances in neural information processing systems, vol. 34, pp. 12 077–12 090, 2021.
[52] J. Yang, L. An, A. Dixit, J. Koo, and S. I. Park, “Depth estimation with simplified transformer,” arXiv preprint arXiv:2204.13791, 2022.
[53] L. Papa, P. Russo, and I. Amerini, “Meter: a mobile vision transformer architecture for monocular depth estimation,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 33, no. 10, pp. 5882–5893, 2023.
[54] J. Bae, K. Hwang, and S. Im, “A study on the generality of neural network structures for monocular depth estimation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023.
[55] F.-A. Croitoru, V. Hondru, R. T. Ionescu, and M. Shah, “Diffusion models in vision: A survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 9, pp. 10 850–10 869, 2023.
[56] T. Amit, T. Shaharbany, E. Nachmani, and L. Wolf, “Segdiff: Image segmentation with diffusion probabilistic models,” arXiv preprint arXiv:2112.00390, 2021.
[57] D. Baranchuk, I. Rubachev, A. Voynov, V. Khrulkov, and A. Babenko, “Label-efficient semantic segmentation with diffusion models,” arXiv preprint arXiv:2112.03126, 2021.
[58] J. Wolleb, R. Sandk¨uhler, F. Bieder, P. Valmaggia, and P. C. Cattin,“Diffusion models for implicit image segmentation ensembles,” in International Conference on Medical Imaging with Deep Learning. PMLR, 2022, pp. 1336–1348.
[59] H. Tan, S. Wu, and J. Pi, “Semantic diffusion network for semantic segmentation,” Advances in Neural Information Processing Systems, vol. 35, pp. 8702–8716, 2022.
[60] T.-Y. Lin, P. Doll´ar, R. Girshick, K. He, B. Hariharan, and S. Belongie,“Feature pyramid networks for object detection,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 2117–2125.

* Levin, Anat, Dani Lischinski, and Yair Weiss. "Colorization using optimization." ACM Transactions on Graphics (ToG). Vol. 23. No. 3. ACM, 2004.
* Silberman, Nathan, et al. "Indoor segmentation and support inference from rgbd images." European Conference on Computer Vision. Springer, Berlin, Heidelberg, 2012.
