# MDS Final Project

## Download Environment and Package
```bash
pip install -r requirements.in
```

## Download Data and Do Preprocessing
```bash
bash download.sh
```
## Start Training!
### Training model with default parameters...
```bash
python train.py
```
### Tune hparameters as you want!
#### for example:
```bash
python train.py --exp_name myexp --time_step 10
```
#### you can check all hyperparameters in opt.py
## HAVE FUN!