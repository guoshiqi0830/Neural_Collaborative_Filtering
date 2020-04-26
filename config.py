DATA_PATH = r'ml-1m/ratings.dat'
MODEL_PATH = r'checkpoint/last_checkpoint.model'

N_USER=6040
N_ITEM=3952 # 3706 

LAYERS=[64,32,16,8]
D_EMBEDDING=8

LR=0.001
BATCH_SIZE=1024
EPOCHS=20

USE_GPU=False

ADAM_BETAS=(0.9, 0.98)
ADAM_EPS=1e-9
ADAM_WEIGHT_DECAY=0
