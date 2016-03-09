#!/usr/bin/env bash
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step_3layers.py -t L3T1Wcustom -r wemb -d 52 -c 2 -u
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step_3layers.py L3T1Wcustom 1

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step_3layers.py -t L3T2Wcustom -r wemb -d 52 -c 4 -u
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step_3layers.py L3T2Wcustom 1

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step_3layers.py -t L3T4Wcustom -r wemb -d 52 -c 7 -u
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step_3layers.py L3T4Wcustom 1

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step_3layers.py -t L3T10Wcustom -r wemb -d 52 -c 18 -u
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step_3layers.py L3T10Wcustom 1

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step_3layers.py -t L3T50Wcustom -r wemb -d 52 -c 86 -u
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step_3layers.py L3T50Wcustom 1

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step_3layers.py -t L3T85Wcustom -r wemb -d 52 -c 150 -u
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step_3layers.py L3T85Wcustom 1

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step.py -t L2T85WrandomTN -r truncnorm -d 52 -c 150 -u
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step.py L2T85WrandomTN 1


