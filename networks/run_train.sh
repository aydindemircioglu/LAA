#!/bin/bash

python3 ./createDataset.py

# mmdetection part
for l in 0.09 0.03 0.009 0.003 0.0009 0.0003
do
  for r in 4 6 8 10 12 14 16 18 20
  do
    python3 ./train.py --model cascadeRCNN --submodel x101 --margin $r --lr $l
    python3 ./predict.py --model cascadeRCNN --submodel x101 --margin $r --lr $l

    python3 ./train.py --model vfnet --submodel x101 --margin $r --lr $l
    python3 ./predict.py --model vfnet --submodel x101 --margin $r --lr $l

    python3 ./train.py --model tood --submodel x101 --margin $r --lr $l
    python3 ./predict.py --model tood --submodel x101 --margin $r --lr $l
  done
done


# yolo part
for n in n s m l x
do
  for m in 4 6 8 10 12 14 16 18 20
  do
    python3 ./yolo.py --modelname=$n --margin=$m --fold=0
  done
done

python3 ./yolo_predict.py

python3 ./modelSelection.py

#
