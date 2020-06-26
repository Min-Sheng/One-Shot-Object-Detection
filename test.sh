CUDA_VISIBLE_DEVICES=0 python test_net.py --dataset 'fss_cell' --net 'res50' \
				   --s 1 --checkepoch 30 --p 192 \
				   --cuda --g 1 --a 5 --k 1 --w --vis

CUDA_VISIBLE_DEVICES=0 python test_net.py --dataset 'fss_cell' --net 'res50' \
				   --s 2 --checkepoch 30 --p 201 \
				   --cuda --g 2 --a 5 --k 1 --w --vis

CUDA_VISIBLE_DEVICES=0 python test_net.py --dataset 'fss_cell' --net 'res50' \
				   --s 3 --checkepoch 30 --p 190 \
				   --cuda --g 3 --a 5 --k 1 --w --vis

CUDA_VISIBLE_DEVICES=0 python test_net.py --dataset 'fss_cell' --net 'res50' \
				   --s 4 --checkepoch 30 --p 190 \
				   --cuda --g 4 --a 5 --k 1 --w --vis

CUDA_VISIBLE_DEVICES=0 python test_net.py --dataset 'fss_cell' --net 'res50' \
				   --s 5 --checkepoch 30 --p 217 \
				   --cuda --g 5 --a 5 --k 1 --w --vis