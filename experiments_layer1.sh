#!/usr/bin/env bash
#3 Layers
eps=("1e-9" "1e-10")
rho=("0.05" "0.25" "0.50" "0.75" "0.95")

for e in "${eps[@]}"
do
for r in "${rho[@]}"
do
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step_3layers.py -t L3A -u adadelta -r "$r" -e "$e"
done
done