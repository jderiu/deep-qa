#!/usr/bin/env bash
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step_3layers.py L3H 1

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step_1layer.py -t L1B -r wemb -d 52 -c 200 -u
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step_layer1.py L1B 1


