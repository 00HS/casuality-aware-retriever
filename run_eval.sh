#Causal Reasoning task
#cawai-dpr
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name=bert-base-cased --model_run_name=cawai_dpr

#cawai-gtr
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name=sentence-transformers/gtr-t5-base --model_run_name=cawai_gtr

#cawai-bge
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name=BAAI/bge-m3-unsupervised  --model_run_name=cawai_bge
