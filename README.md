# MPHIL
Code for MPHIL, which partially inherits from the code of https://github.com/HICAI-ZJU/iMoLD


This repo is also depended on `GOOD` and `DrugOOD`, please follow the installation methods provided for each package:
- GOOD (Version 1.1.1)
  - Repository: https://github.com/divelab/GOOD/
  - Installation: Please follow the instructions provided in the repository to install.
- DrugOOD (Version 0.0.1)
  - Repository: https://github.com/tencent-ailab/DrugOOD
  - Installation: Please follow the instructions provided in the repository to install.


## Data
The data used in the experiments can be downloaded from the following sources:

1. GOOD
   - [GOODPCBA](https://drive.google.com/file/d/1WGieOjtgNXtGoO6o1EGhKrZj0zWU7AJl/view?usp=sharing)
   - [GOODHIV](https://drive.google.com/file/d/1CoOqYCuLObnG5M0D8a2P2NyL61WjbCzo/view?usp=sharing)
   - [GOODZINC](https://drive.google.com/file/d/1CHR0I1JcNoBqrqFicAZVKU3213hbsEPZ/view?usp=sharing)
   - Extract the downloaded files and save the contents in the `data` directory.
2. DrugOOD
    - download from [link](https://drive.google.com/drive/folders/19EAVkhJg0AgMx7X-bXGOhD4ENLfxJMWC).
    - Extract the downloaded file and save the contents in the `drugood-data-chembl30` directory.

An example of the folder hierarchy after adding the data files:
```
├── data
│   ├── GOODHIV
│   ├── GOODPCBA
│   ├── GOODZINC
├── drugood-data-chembl30
│   ├── lbap_core_ec50_assay.json
│   └── ...
├── models
│   ├── model.py
│   └── ...
├── run.py
└── README.md

```

## Running Script
#### Training
```
python run.py --dataset GOODZINC --domain scaffold --shift concept 
```
Running parameters and descriptions are as follows:
| Parameter | Description | Choices |
| --- | --- | --- |
| dataset | name of dataset | `GOODHIV`, `GOODZINC`, `GOODPCBA`, `ic50_assay`, `ic50_scaffold`, `ic50_size`, `ec50_assay`, `ec50_scaffold`, `ec50_size`.|
| domain | environment-splitting strategy | `scaffold`, `size`. Only need to be specified for datasets in `GOOD`. |
| shift | type of distribution shift | `covariate`, `concept`. Only need to be specified for datasets in `GOOD`. |

