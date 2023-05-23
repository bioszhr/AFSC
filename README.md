# AFSC: A self-supervised Augmentation-Free Spatial Clustering method based on contrastive learning for spatial transcriptomics
## Overview
We propose a new self-supervised spatial clustering method, Augmentation-Free Spatial Clustering (AFSC), which integrates spatial information and gene expression to learn latent representations without negative pairs or data augmentation. We also introduce a clustering loss to guide the training along with the contrastive loss. Experiments on multiple datasets demonstrate that our method outperforms existing methods in self-supervised spatial clustering tasks. Furthermore, the learned representations can be used for various downstream tasks, including clustering, visualization, and trajectory inference.<br>
![image]()
## Requirements
Please ensure that all the libraries below are successfully installed:<br>
-Python 3.7<br>
-torch 1.12.0<br>
-torch-geometric 1.7.0<br>
-faiss 1.7.0<br>
## Run AFSC on the example data.
If you wanna run datasets at spot resolution, e.g. the breast invasive carcinoma (BRCA), you should put the file "filtered_feature_bc_matrix.h5" and "metadata.tsv" in ***datasets/spot***, and run ***spot.ipynb***.<br>
If you wanna run datasets at single-cell resolution, e.g. the mouse hypothalamic preoptic area obtained with MERFISH, you should put the file "Moffitt_and_Bambah-Mukku_et_al_merfish_all_cells.csv" in ***datasets/single-cell***, and run ***single-cell.ipynb***.
### Output
The output results will be stored in the dir ***results***.
### Datasets
The spatial transcriptomics datasets utilized in this study can be downloaded from: (1) the human dorsolateral prefrontal cortex (DLPFC): http://research.libd.org/spatialLIBD/. (2) the breast invasive carcinoma (BRCA): https://www.10xgenomics.com/resources/datasets. (3) the anterior of the mouse brain tissues (MBA): https://www.10xgenomics.com/resources/datasets. (4) the mouse hypothalamic preoptic area dataset obtained with MERFISH: https://datadryad.org/stash/dataset/doi:10.5061/dryad.8t8s248. (5) the mouse cortex subventricular zone (cortex_SVZ) and mouse olfactory bulb (OB) independent tissues obtained with seqFISH+: https://github.com/CaiGroup/seqFISH-PLUS.
