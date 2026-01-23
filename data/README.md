## Data Preparation

### Dataset
The dataset used in this project is the **VinDR-SpineXR** dataset, which can be downloaded from
[VinDR-SpineXR on PhysioNet](https://physionet.org/content/vindr-spinexr/1.0.0/)

After downloading and extracting the dataset, organize the data directory as follows:

```text
project_home/
└── data/
    └── dicom_dataset/
        ├── annotations/
        │   ├── train.csv
        │   └── test.csv
        ├── train_images/
        │   ├── 000f985efcb28afd281e3cd1b4d370ee.dicom
        │   ├── ...
        │   └── fffcdf23fd9958fd802de0ae99a1823f.dicom
        └── test_images/
            ├── 000b3dad09378f680c845f8d7827d6ad.dicom
            ├── ...
            └── ffec804215c7a442c4b480c9c4a2c5c8.dicom
```

## Convert DICOM to PNG
Run the following bash script. Note that you might need to change the number of CPU workers and debug mode
```bash
convert_dicom.bat
```
Data folder structure after converting:
```text
project_home/
└── data/
    ├── dicom_dataset/
    │   ├── annotations/
    │   │   ├── train.csv
    │   │   └── test.csv
    │   ├── train_images/
    │   └── test_images/
    ├── train_pngs/
    │   ├── 0e2f7b5a29c858128c7a07384d71b116.png
    │   ├── ...
    │   └── f4f7b2b74e0dfa38fbb31abdc2458eb0.png
    └── test_pngs/
        ├── 2d7872d7c49ea95849f38ad035690c8d.png
        ├── ...
        └── ff6a81f9fa386401ce11a0eb74e1f661.png
```
## Preprocessing Data
To preprocessing data, run the script:
```bash
preprocess.bat
```
