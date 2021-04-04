import time

from easydict import EasyDoct as edict

## Init
__C = edict()
cfg = __C

## seed Value
__C.SEED = 2021

## datasets: X, Y
__C.DATASET = None

## networks: X, Y
__C.NET = None

## gpu id
__C.GPU_ID = [0]

## learning rate
__C.LR = 0
# LR scheduler

## training settings
__C.MAX_EPOCH = 0

## experiment
now = time.strftime('%m-%d_%H-%M', time.localtime())
__C.EXP_NAME = now \
    + '_' + __C.DATASET \
    + '_' + __C.NET

## loggin
__C.EXP_PATH = '/experiments'
