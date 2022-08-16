# BiMaL: Bijective Maximum Likelihood Approach to Domain Adaptation in Semantic Scene Segmentation

## Install

This repo requires Python 3.6+, Pytorch >= 0.4.1, and CUDA 9.0+.

```bash
git clone https://github.com/uark-cviu/BiMaL
cd BiMaL
pip install -e ADVENT
```

## Training

```bash
cd ADVENT/advent/scripts
python train.py --cfg nvp_config/gta2cityscape_advent_nvp.yaml
```

## Datasets

* **GTA5**: Please follow the instructions [here](https://download.visinf.tu-darmstadt.de/data/from_games/) to download images and semantic segmentation annotations. The GTA5 dataset directory should have this basic structure:
```bash
<root_dir>/data/GTA5/                               % GTA dataset root
<root_dir>/data/GTA5/images/                        % GTA images
<root_dir>/data/GTA5/labels/                        % Semantic segmentation labels
...
```

* **Cityscapes**: Please follow the instructions in [Cityscape](https://www.cityscapes-dataset.com/) to download the images and validation ground-truths. The Cityscapes dataset directory should have this basic structure:
```bash
<root_dir>/data/Cityscapes/                         % Cityscapes dataset root
<root_dir>/data/Cityscapes/leftImg8bit              % Cityscapes images
<root_dir>/data/Cityscapes/leftImg8bit/val
<root_dir>/data/Cityscapes/gtFine                   % Semantic segmentation labels
<root_dir>/data/Cityscapes/gtFine/val
...
```


## Acknowledgements
This codebase is heavily borrowed from [ADVENT](https://github.com/valeoai/ADVENT).

## Citation

If you find this code useful for your research, please consider citing:
```
@inproceedings{truong2021bimal,
  title={BiMaL: Bijective Maximum Likelihood Approach to Domain Adaptation in Semantic Scene Segmentation},
  author={Truong, Thanh-Dat and Duong, Chi Nhan and Le, Ngan and Phung, Son Lam and Rainwater, Chase and Luu, Khoa},
  booktitle={International Conference on Computer Vision},
  year={2021}
}
```
