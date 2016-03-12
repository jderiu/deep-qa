#!/usr/bin/env bash
#2 Layer
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step.py -t L2T0Wcustom -r wemb -d 52 -c 0 -u
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step.py L2T0Wcustom 1

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step.py -t L2T1Wcustom -r wemb -d 52 -c 2 -u
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step.py L2T1Wcustom 1

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step.py -t L2T2Wcustom -r wemb -d 52 -c 4 -u
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step.py L2T2Wcustom 1

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step.py -t L2T4Wcustom -r wemb -d 52 -c 7 -u
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step.py L2T4Wcustom 1

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step.py -t L2T10Wcustom -r wemb -d 52 -c 18 -u
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step.py L2T10Wcustom 1

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step.py -t L2T50Wcustom -r wemb -d 52 -c 86 -u
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step.py L2T50Wcustom 1

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step.py -t L2T85Wcustom -r wemb -d 52 -c 150 -u
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step.py L2T85Wcustom 1