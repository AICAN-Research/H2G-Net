# H2G-Net: A multi-resolution approach for semantic segmentation of gigapixel histopathological images
This repository contains code relevant for the proposed design H2G-Net. The architecture is a cascaded convolutional neural network which is divided into two stages. The first stage, the detection stage, applies a patch-wise classifier across the image and produces a heatmap. In the second stage, the refinement stage, the heatmap concatenated with a low-resolution version of the original WSI is propagated through a fully-connected convolutional autoencoder that produces a final segmentation of the tumour ROI.

## Setup

## Something...

## Citation
Please, cite our paper if you find the work useful:
<pre>
  @MISC{pedersen2021H2GNet,
  title={Hybrid-guiding: a multi-resolution approach for semantic segmentation of gigapixel histopathological images},
  author={André Pedersen, Erik Smistad, Anna M. Bofin, Tor V. Rise, Vibeke G. Dale, Henrik S. Pettersen, David Bouget, Tor-Arne S. Nordmo, Ingerid Reinertsen, Marit Valla},
  year={2021},
  eprint={some.numbers},
  archivePrefix={arXiv},
  primaryClass={eess.IV}}
</pre>

## Contact
Please, contact andre.pedersen@ntnu.no for any further questions.

## Acknowledgements
Code for the AGU-Net and DAGU-Net architectures were based on the publication:
<pre>
  @MISC{bouget2021meningioma,
  title={Meningioma segmentation in T1-weighted MRI leveraging global context and attention mechanisms},
  author={David Bouget and André Pedersen and Sayied Abdol Mohieb Hosainey and Ole Solheim and Ingerid Reinertsen},
  year={2021},
  eprint={2101.07715},
  archivePrefix={arXiv},
  primaryClass={eess.IV}}
</pre>

Code for the DoubleU-Net architectures were based on the official GitHub [repository](https://github.com/DebeshJha/2020-CBMS-DoubleU-Net), based on this publication:
<pre>
  @INPROCEEDINGS{9183321,
  author={D. {Jha} and M. A. {Riegler} and D. {Johansen} and P. {Halvorsen} and H. D. {Johansen}},
  booktitle={2020 IEEE 33rd International Symposium on Computer-Based Medical Systems (CBMS)}, 
  title={DoubleU-Net: A Deep Convolutional Neural Network for Medical Image Segmentation}, 
  year={2020},
  pages={558-564}}
</pre>
