CUDA_VISIBLE_DEVICES=0 python test_net.py --dataset 'fss_cell' --net 'res50' \
				   --s 1 --checkepoch 30 --p 98 \
				   --cuda --g 1 --vis

CUDA_VISIBLE_DEVICES=0 python test_net.py --dataset 'fss_cell' --net 'res50' \
				   --s 2 --checkepoch 30 --p 101 \
				   --cuda --g 2 --vis

CUDA_VISIBLE_DEVICES=0 python test_net.py --dataset 'fss_cell' --net 'res50' \
				   --s 3 --checkepoch 30 --p 89 \
				   --cuda --g 3 --vis

CUDA_VISIBLE_DEVICES=0 python test_net.py --dataset 'fss_cell' --net 'res50' \
				   --s 4 --checkepoch 30 --p 100 \
				   --cuda --g 4 --vis

CUDA_VISIBLE_DEVICES=0 python test_net.py --dataset 'fss_cell' --net 'res50' \
				   --s 5 --checkepoch 30 --p 106 \
				   --cuda --g 5 --vis