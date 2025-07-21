# H2G-Net

[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://zenodo.org/badge/DOI/10.3389/fneur.2022.932219.svg)](https://doi.org/10.3389/fmed.2022.971873)
[![Poster](https://img.shields.io/badge/Poster-PDF-f39f37)](https://github.com/andreped/H2G-Net/blob/main/poster/poster.pdf)

This repository contains the code relevant for the proposed design H2G-Net, which was introduced in the manuscript [*"H2G-Net: A multi-resolution refinement approach for segmentation of breast cancer region in gigapixel histopathological images"*](https://www.frontiersin.org/articles/10.3389/fmed.2022.971873/full), published in Frontiers in Medicine.

The work was also presented at a region conference (HMN RHF 2022), where it won best poster award!

## Brief summary of the paper and contributions

We propose a cascaded convolutional neural network for semantic segmentation of breast cancer tumours from whole slide images (WSIs). It is a two-stage design. In the first stage (detection stage), we apply a patch-wise classifier across the image which produces a tumour probability heatmap. In the second stage (refinement stage), we merge the resultant heatmap with a low-resolution version of the original WSI, before we send it to a new convolutional autoencoder that produces a final segmentation of the tumour ROI.

- The paper proposed a hierarchically-balanced sampling scheme to adjust for the many data imbalance problems:

<p align="center">
<img src="https://user-images.githubusercontent.com/29090665/190863187-a239afc5-7a98-48df-9b5e-f0bd899b2d76.jpg" width="80%">
</p>

- Second, a two-stage cascaded convolutional neural network design, H2G-Net, was proposed that utilizes a refinement network to refine generated patch-wise predictions to improve low-resolution segmentation of breast cancer region.

<p align="center">
<img src="https://user-images.githubusercontent.com/29090665/190863086-ced55fbb-b4ed-4b4e-be56-3b9c4b6d474d.jpg" width="80%">
</p>

- The final model has been integrated into the open software [FastPathology](https://github.com/AICAN-Research/FAST-Pathology) and only takes ~1 minute to use on a full whole slide image using the CPU.

- Developed annotated dataset of 624 whole slide images of breast cancer.

## Test the model on your own data
You can easily test the H2G-Net model using [FAST](https://fast.eriksmistad.no).
First make sure you have [all requirements for FAST installed](https://fast.eriksmistad.no/install.html).
Then install FAST using pip, and run the breast tumour segmentation pipeline from your terminal.
This will download the model, run it on the WSI you specify and visualize the results.
```bash
pip install pyFAST
runPipeline --datahub breast-tumour-segmentation --file path/to/your/whole-slide-image.vsi
```

Or you can test the model in the graphical user interface [FastPathology](https://github.com/AICAN-Research/FAST-Pathology) which allows you to run the model on multiple images, change the visualization of the segmentation and export the segmentation to disk.

## Code info
Other useful scripts and tips for importing/exporting predictions/annotations to/from QuPath <-> FastPathology can be found in the [NoCodeSeg](https://github.com/andreped/NoCodeSeg) repository.

**Disclaimer:** The source code is provided as is, only to demonstrate how to define the architecture and design used in the paper. The code itself requires modifications to run on a new dataset, as it contains various hard-coded solutions, but all components are provided, as well as the code for training and evaluating the refinement model.


## How to cite
Please, cite our paper if you find the work useful:
<pre>
@article{10.3389/fmed.2022.971873,
  author={Pedersen, André and Smistad, Erik and Rise, Tor V. and Dale, Vibeke G. and Pettersen, Henrik S. and Nordmo, Tor-Arne S. and Bouget, David and Reinertsen, Ingerid and Valla, Marit},
  title={H2G-Net: A multi-resolution refinement approach for segmentation of breast cancer region in gigapixel histopathological images},
  journal={Frontiers in Medicine},
  volume={9},
  year={2022},
  url={https://www.frontiersin.org/articles/10.3389/fmed.2022.971873},
  doi={10.3389/fmed.2022.971873},
  issn={2296-858X}
}
</pre>

## Contact
Please, contact andrped94@gmail.com for any further questions.

## Acknowledgements
Code for the AGU-Net and DAGU-Net architectures were based on the publication:
<pre>
@misc{bouget2021meningioma,
  title={Meningioma segmentation in T1-weighted MRI leveraging global context and attention mechanisms},
  author={David Bouget and André Pedersen and Sayied Abdol Mohieb Hosainey and Ole Solheim and Ingerid Reinertsen},
  year={2021},
  eprint={2101.07715},
  archivePrefix={arXiv},
  primaryClass={eess.IV}
}
</pre>

Code for the DoubleU-Net architectures were based on the official GitHub [repository](https://github.com/DebeshJha/2020-CBMS-DoubleU-Net), based on this publication:
<pre>
@INPROCEEDINGS{9183321,
  author={D. {Jha} and M. A. {Riegler} and D. {Johansen} and P. {Halvorsen} and H. D. {Johansen}},
  booktitle={2020 IEEE 33rd International Symposium on Computer-Based Medical Systems (CBMS)}, 
  title={DoubleU-Net: A Deep Convolutional Neural Network for Medical Image Segmentation}, 
  year={2020},
  pages={558-564}
}
</pre>
