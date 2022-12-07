# MDS Final Project

## Clone Project
```bash
git clone https://github.com/qqandy0120/MDS-Final-Project.git
cd MDS-Final-Project
```

## [Optinal] Create Conda Environment
```bash
conda create -n MDSenv python=3.8
conda activate MDSenv
```

## Install Package
```bash
pip install -r requirements.in
```

## Download Data and Do Preprocessing
```bash
bash download.sh
```
### if **wget: command not found**, try:
```bash
conda install wget
```
and do
```bash
bash download.sh
```
again

### or you can just download data from [here](https://www.dropbox.com/s/oim8d8dnl2r3p8s/Flotation_Plant_preprocessed.csv?dl=0)
### and put it in ./MDS-Final-Project/data/Flotation_Plant_preprocessed.csv 
then do
```bash
python preprocess.py
```

## Start Training!
### Training model with default parameters...
```bash
python train.py
```
### or Tune hparameters as you want!
#### for example:
```bash
python train.py --exp_name myexp --time_step 10
```
#### you can check all hyperparameters in [opt.py](https://github.com/qqandy0120/MDS-Final-Project/blob/main/opt.py)
## HAVE FUN!
