#Causal Reasoning task
#cawai-dpr
python model/cawai_dpr/train.py --model_name=bert-base-cased --model_run_name=cawai_dpr

#cawai-gtr
python model/cawai_gtr/train.py --model_name=sentence-transformers/gtr-t5-base --model_run_name=cawai_gtr

#cawai-bge
python model/cawai_gtr/train.py --model_name=BAAI/bge-m3-unsupervised  --model_run_name=cawai_bge

#CausalQA task
#cawai-dpr
python model/cawai_dpr/train.py --model_name=bert-base-cased --model_run_name=cawai_dpr_causalqa --config_path=config_cq.yaml

#cawai-gtr
python model/cawai_gtr/train.py --model_name=sentence-transformers/gtr-t5-base --model_run_name=cawai_gtr_causalqa --config_path=config_cq.yaml

#cawai-bge
python model/cawai_gtr/train.py --model_name=BAAI/bge-m3-unsupervised  --model_run_name=cawai_bge_causalqa --config_path=config_cq.yaml
