
#!/bin/bash

export PYTHONIOENCODING=UTF-8
export PYTHONPATH=./:$PYTHONPATH

#export CUDA_VISIBLE_DEVICES="0,1"
#export LOCAL_RANK="0"
# The number of gpus runing on each node/machine
#num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

#log_timestamp=`date +%Y%m%d%T | sed 's/://g'`
#log_file=${out_dir}/log_${log_timestamp}

config_file=./config/train_macbert_lm.yaml
ckpt_model=workspace_macbert/model/01macbert_model.ep0_80673.pt

	#--nnodes=2 --node_rank=1 \
	#--master_addr="192.168.3.33" \
	#--master_port=5324 \
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 \
#	bin/dist_train_macbert_lm.py --config ${config_file}

	#bin/dist_train_macbert_lm.py --config ${config_file} --check_point ${ckpt_model}
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 \
	bin/dist_train_macbert_lm.py --config ${config_file}




