# H2G-Net

⚠️***Latest: Won best poster award at HMN RHF 2022 conference!***

This repository contains the code relevant for the proposed design H2G-Net, which was introduced in the manuscript [*"Hybrid guiding: A multi-resolution refinement approach for semantic segmentation of gigapixel histopathological images"*](https://arxiv.org/abs/2112.03455). 

We propose a cascaded convolutional neural network for semantic segmentation of breast cancer tumours from whole slide images (WSIs). It is a two-stage design. In the first stage (detection stage), we apply a patch-wise classifier across the image which produces a tumour probability heatmap. In the second stage (refinement stage), we merge the resultant heatmap with a low-resolution version of the original WSI, before we send it to a new convolutional autoencoder that produces a final segmentation of the tumour ROI.

**NOTE: This repository is currently in construction! More to be added!!**

## Citation
Please, cite our paper if you find the work useful:
<pre>
  @misc{pedersen2021hybrid,
  title={Hybrid guiding: A multi-resolution refinement approach for semantic segmentation of gigapixel histopathological images}, 
  author={André Pedersen and Erik Smistad and Tor V. Rise and Vibeke G. Dale and Henrik S. Pettersen and Tor-Arne S. Nordmo and David Bouget and Ingerid Reinertsen and Marit Valla},
  year={2021},
  eprint={2112.03455},
  archivePrefix={arXiv},
  primaryClass={eess.IV}}
</pre>

## Contact
Please, contact andre.pedersen@ntnu.no for any further questions.

## Acknowledgements
Code for the AGU-Net and DAGU-Net architectures were based on the publication:
<pre>
  @misc{bouget2021meningioma,
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
