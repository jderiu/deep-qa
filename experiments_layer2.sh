#!/usr/bin/env bash
#3 Layers
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step.py -t L2T0WcustomKmax -r wemb -d 52 -c 0 -u
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step.py L2T0WcustomKmax 1

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step.py -t L3T4WcustomKmax -r wemb -d 52 -c 7 -u
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step.py L3T4WcustomKmax 1


THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step.py -t L3T85WcustomKmax -r wemb -d 52 -c 150 -u
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step.py L3T85WcustomKmax 1



