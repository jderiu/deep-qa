#!/usr/bin/env bash

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step.py -t L2T0Wrandomtrue -r uniform -d 52 -c 0 -u
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step.py L2T0Wrandomtrue 1

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step.py -t L2T0Wrandomfalse -r uniform -d 52 -c 0
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step.py L2T0Wrandomfalse 1

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step.py -t L2T4Wrandomtrue -r uniform -d 52 -c 7 -u
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step.py L2T4Wrandomtrue 1

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step.py -t L2T4Wrandomfalse -r uniform -d 52 -c 7
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step.py L2T4Wrandomfalse 1

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step.py -t L2T85Wrandomtrue -r uniform -d 52 -c 150 -u
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step.py L2T85Wrandomtrue 1

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step.py -t L2T85Wrandomfalse -r uniform -d 52 -c 150
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step.py L2T85Wrandomfalse 1