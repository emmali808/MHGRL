#export conv='gcn'
#export conv='gat'
#export conv='rgcn'
#export conv='rgat'
#export conv='gps'
export conv='anti'

export epoch=30

export layer=2
export hidden_size=250
export pair_neurons=25

export task='cluster'
export train_data='preprocess_data/mimic3/train_cluster.pt'
# export task='knn'
# export train_data='preprocess_data/mimic3/train_knn.pt'

export train_input_data='preprocess_data/mimic3/train_input_knn.pt'
export valid_cluster_data='preprocess_data/mimic3/valid_cluster.pt'
export valid_knn_data='preprocess_data/mimic3/valid_knn.pt'
export test_cluster_data='preprocess_data/mimic3/test_cluster.pt'
export test_knn_data='preprocess_data/mimic3/test_knn.pt'
export output_dir='res/mimic3/'
export lr=0.0001


CUDA_VISIBLE_DEVICES=0 python train.py  --resume_path "/home/liufeiyan/EHRModel/MHGRL/code/res/mimic3/pytorch_cluster.bin" --do_test --dataset mimic3 --task $task  --use_conv $conv --gcn_conv_nums $layer --hidden_size $hidden_size --pair_neurons $pair_neurons --batch_size 256 --epoch $epoch --learning_rate $lr --valid_cluster_data $valid_cluster_data --valid_knn_data $valid_knn_data --test_knn_data $test_knn_data --train_data $train_data --test_cluster_data $test_cluster_data --output_dir $output_dir --do_train --train_input_data $train_input_data