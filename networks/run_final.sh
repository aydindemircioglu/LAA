#!/bin/bash

# here be dragons. best model from cross-validation
# (since yolo was not best, this code is for mmdetection only)
model=vfnet
submodel=x101
margin=14
lr=0.003

# retrain model
python3 ./train.py --model $model --submodel $submodel --margin $margin --final --lr $lr
python3 ./predict.py --model $model --submodel $submodel --margin $margin --lr $lr --final --testset val
python3 ./predict.py --model $model --submodel $submodel --margin $margin --lr $lr --final --testset test

#
