#!/usr/bin/env bash
#3 Layers
eps=("1e-1" "1e-2" "1e-3" "1e-4" "1e-5" "1e-6" "1e-7" "1e-8")
rho=("0.05" "0.25")

for e in "${eps[@]}"
do
for r in "${rho[@]}"
do
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step_3layers.py -t L3A -u adadelta -r "$r" -e "$e"
done
done

