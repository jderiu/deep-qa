#!/usr/bin/env bash
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step_1layer.py L1T0Wcustomtrue_1 0 False False
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step_layer1.py L1T0Wcustomtrue_1 0 False False

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step_1layer.py L1T1Wcustomtrue_1 2 False False
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step_layer1.py L1T1Wcustomtrue_1 2 False False

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step_1layer.py L1T2Wcustomtrue_1 4 False False
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step_layer1.py L1T2Wcustomtrue_1 4 False False

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step_1layer.py L1T4Wcustomtrue_1 7 False False
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step_layer1.py L1T4Wcustomtrue_1 7 False False

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step_1layer.py L1T10Wcustomtrue_1 18 False False
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step_layer1.py L1T10Wcustomtrue_1 12 False False

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step_1layer.py L1T50Wcustomtrue_1 86 False False
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step_layer1.py L1T50Wcustomtrue_1 12 False False

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step_1layer.py L1T85Wcustomtrue_1 150 False False
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step_layer1.py L1T85Wcustomtrue_1 12 False False

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step.py L2T0Wcustomtrue_1 0 False False
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step.py L2T0Wcustomtrue_1 0 False False

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step.py L2T1Wcustomtrue_1 2 False False
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step.py L2T1Wcustomtrue_1 2 False False

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step.py L2T2Wcustomtrue_1 4 False False
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step.py L2T2Wcustomtrue_1 4 False False

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step.py L2T4Wcustomtrue_1 7 False False
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step.py L2T4Wcustomtrue_1 7 False False

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step.py L2T10Wcustomtrue_1 18 False False
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step.py L2T10Wcustomtrue_1 12 False False

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step.py L2T50Wcustomtrue_1 86 False False
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step.py L2T50Wcustomtrue_1 12 False False

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step.py L2T85Wcustomtrue_1 150 False False
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step.py L2T85Wcustomtrue_1 12 False False


THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step_1layer.py L1T0Wcustomtrue_2 0 False False
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step_layer1.py L1T0Wcustomtrue_2 0 False False

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step_1layer.py L1T1Wcustomtrue_2 2 False False
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step_layer1.py L1T1Wcustomtrue_2 2 False False

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step_1layer.py L1T2Wcustomtrue_2 4 False False
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step_layer1.py L1T2Wcustomtrue_2 4 False False

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step_1layer.py L1T4Wcustomtrue_2 7 False False
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step_layer1.py L1T4Wcustomtrue_2 7 False False

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step_1layer.py L1T10Wcustomtrue_2 18 False False
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step_layer1.py L1T10Wcustomtrue_2 12 False False

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step_1layer.py L1T50Wcustomtrue_2 86 False False
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step_layer1.py L1T50Wcustomtrue_2 12 False False

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step_1layer.py L1T85Wcustomtrue_2 150 False False
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step_layer1.py L1T85Wcustomtrue_2 12 False False

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step.py L2T0Wcustomtrue_2 0 False False
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step.py L2T0Wcustomtrue_2 0 False False

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step.py L2T1Wcustomtrue_2 2 False False
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step.py L2T1Wcustomtrue_2 2 False False

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step.py L2T2Wcustomtrue_2 4 False False
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step.py L2T2Wcustomtrue_2 4 False False

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step.py L2T4Wcustomtrue_2 7 False False
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step.py L2T4Wcustomtrue_2 7 False False

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step.py L2T10Wcustomtrue_2 18 False False
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step.py L2T10Wcustomtrue_2 12 False False

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step.py L2T50Wcustomtrue_2 86 False False
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step.py L2T50Wcustomtrue_2 12 False False

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python distant_supervised_step.py L2T85Wcustomtrue_2 150 False False
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python supervised_step.py L2T85Wcustomtrue_2 12 False False