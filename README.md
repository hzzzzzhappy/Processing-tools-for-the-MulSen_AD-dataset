# Processing-tools-for-the-MulSen_AD-dataset
😊 This repository contains Processing tools for the MulSen_AD Datasets from ["Multi-Sensor Object Anomaly Detection: Unifying Appearance, Geometry, and Internal Properties"](https://github.com/ZZZBBBZZZ/MulSen-AD/tree/main). The aim is to convert the supplied data into a format similar to Real3D-AD and Anomaly-ShapeNet. 

This library is suitable for pure 3D anomaly detection. This library will reorganise the data as follows:
```bash
MulSen_AD_processed/
├── capsule/
│   ├── train/
│   │   ├── 0_good.pcd
│   │   ├── 1_good.pcd
│   │   └── ...
│   ├── test/
│   │   ├── 0_bad.pcd
│   │   ├── 1_bad.pcd
│   │   ├── 2_good.pcd
│   │   └── ...
│   └── GT/
│       ├── 0_bad.txt
│       ├── 1_bad.txt
│       └── ...
├── cotton/
│   ├── train/
│   ├── test/
│   └── GT/
└── ...
```
It will generate a MulSen_AD_processed directly in the sibling directory of the MuSen_AD. This format can be directly loaded by dataload directly from papers such as [Reg3D-AD](https://github.com/m-3lab/real3d-ad), [ISMP](https://github.com/M-3LAB/Look-Inside-for-More), [MC3D-AD](https://github.com/jiayi-art/MC3D-AD) and [MC4AD](https://github.com/hzzzzzhappy/MC4AD). 

The original format of MulSen_AD is as follows:
```bash
MulSen_AD
├── capsule                              ---Object class folder.
    ├── RGB                              ---RGB images
        ├── train                        ---A training set of RGB images
            ├── 0.png
            ...
        ├── test                         ---A test set of RGB images
            ├── hole                     ---Types of anomalies, such as hole. 
                ├── 0.png
                ...
            ├── crack                    ---Types of anomalies, such as crack.
                ├── 0.png
                ...
            ├── good                     ---RGB images without anomalies.
                ├── 0.png
                ...
            ...
        ├── GT                           ---GT segmentation mask for various kinds of anomalies.
            ├── hole
                ├── 0.png
                ├── data.csv             ---Label information
                ...
            ├── crack
                ├── 0.png
                ├── data.csv
                ...
            ├── good
                ├── data.csv
            ...
        ...
    ├── Infrared                        ---Infrared images
        ├── train
        ├── test
        ├── GT
    ├── Pointcloud                      ---Point Clouds
        ├── train
        ├── test
        ├── GT
├── cotton                             ---Object class folder.                      
    ... 
...
```

## How to Process
Download our process.py in the MuSen_AD folder. Run:
```bash
python process.py
```



## 😊 If this helps you, I'm delighted.
