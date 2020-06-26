CUDA_VISIBLE_DEVICES=0 python trainval_net.py \
				   --dataset 'fss_cell' --net 'res50' \
				   --bs 8 --nw 8 \
				   --lr 0.001 --lr_decay_step 10 \
				   --cuda --s 1 --g 1 --seen 1 --epochs 30 --k 1

CUDA_VISIBLE_DEVICES=0 python trainval_net.py \
				   --dataset 'fss_cell' --net 'res50' \
				   --bs 8 --nw 8 \
				   --lr 0.001 --lr_decay_step 10 \
				   --cuda --s 2 --g 2 --seen 1 --epochs 30 --k 1

CUDA_VISIBLE_DEVICES=0 python trainval_net.py \
				   --dataset 'fss_cell' --net 'res50' \
				   --bs 8 --nw 8 \
				   --lr 0.001 --lr_decay_step 10 \
				   --cuda --s 3 --g 3 --seen 1 --epochs 30 --k 1

CUDA_VISIBLE_DEVICES=0 python trainval_net.py \
				   --dataset 'fss_cell' --net 'res50' \
				   --bs 8 --nw 8 \
				   --lr 0.001 --lr_decay_step 10 \
				   --cuda --s 4 --g 4 --seen 1 --epochs 30 --k 1

CUDA_VISIBLE_DEVICES=0 python trainval_net.py \
				   --dataset 'fss_cell' --net 'res50' \
				   --bs 8 --nw 8 \
				   --lr 0.001 --lr_decay_step 10 \
				   --cuda --s 5 --g 5 --seen 1 --epochs 30 --k 1
