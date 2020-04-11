#!/bin/bash


source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/cnn_autoencoder/start_feature_maps_2/growth_factor_2/data_augmentation_0/ --name n --type cnn_autoencoder --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_autoencoder_starting_feature_map 2 --cnn_autoencoder_growth_factor 2 --cnn_autoencoder_kernel_size 3
source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/cnn_autoencoder/start_feature_maps_2/growth_factor_3/data_augmentation_0/ --name n --type cnn_autoencoder --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_autoencoder_starting_feature_map 2 --cnn_autoencoder_growth_factor 3 --cnn_autoencoder_kernel_size 3
source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/cnn_autoencoder/start_feature_maps_4/growth_factor_2/data_augmentation_0/ --name n --type cnn_autoencoder --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_autoencoder_starting_feature_map 4 --cnn_autoencoder_growth_factor 2 --cnn_autoencoder_kernel_size 3
source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/cnn_autoencoder/start_feature_maps_4/growth_factor_3/data_augmentation_0/ --name n --type cnn_autoencoder --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_autoencoder_starting_feature_map 4 --cnn_autoencoder_growth_factor 3 --cnn_autoencoder_kernel_size 3
source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/cnn_autoencoder/start_feature_maps_8/growth_factor_2/data_augmentation_0/ --name n --type cnn_autoencoder --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_autoencoder_starting_feature_map 8 --cnn_autoencoder_growth_factor 2 --cnn_autoencoder_kernel_size 3
source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/cnn_autoencoder/start_feature_maps_8/growth_factor_3/data_augmentation_0/ --name n --type cnn_autoencoder --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_autoencoder_starting_feature_map 8 --cnn_autoencoder_growth_factor 3 --cnn_autoencoder_kernel_size 3
source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/cnn_autoencoder/start_feature_maps_2/growth_factor_2/data_augmentation_1/ --name n --type cnn_autoencoder --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_autoencoder_starting_feature_map 2 --cnn_autoencoder_growth_factor 2 --cnn_autoencoder_kernel_size 3
source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/cnn_autoencoder/start_feature_maps_2/growth_factor_3/data_augmentation_1/ --name n --type cnn_autoencoder --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_autoencoder_starting_feature_map 2 --cnn_autoencoder_growth_factor 3 --cnn_autoencoder_kernel_size 3
source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/cnn_autoencoder/start_feature_maps_4/growth_factor_2/data_augmentation_1/ --name n --type cnn_autoencoder --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_autoencoder_starting_feature_map 4 --cnn_autoencoder_growth_factor 2 --cnn_autoencoder_kernel_size 3
source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/cnn_autoencoder/start_feature_maps_4/growth_factor_3/data_augmentation_1/ --name n --type cnn_autoencoder --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_autoencoder_starting_feature_map 4 --cnn_autoencoder_growth_factor 3 --cnn_autoencoder_kernel_size 3
source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/cnn_autoencoder/start_feature_maps_8/growth_factor_2/data_augmentation_1/ --name n --type cnn_autoencoder --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_autoencoder_starting_feature_map 8 --cnn_autoencoder_growth_factor 2 --cnn_autoencoder_kernel_size 3
source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/cnn_autoencoder/start_feature_maps_8/growth_factor_3/data_augmentation_1/ --name n --type cnn_autoencoder --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_autoencoder_starting_feature_map 8 --cnn_autoencoder_growth_factor 3 --cnn_autoencoder_kernel_size 3


source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/cnn_vae/start_feature_maps_2/growth_factor_2/data_augmentation_0/ --name n --type cnn_vae --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_vae_starting_feature_map 2 --cnn_vae_growth_factor 2 --cnn_vae_kernel_size 3
source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/cnn_vae/start_feature_maps_2/growth_factor_3/data_augmentation_0/ --name n --type cnn_vae --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_vae_starting_feature_map 2 --cnn_vae_growth_factor 3 --cnn_vae_kernel_size 3
source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/cnn_vae/start_feature_maps_4/growth_factor_2/data_augmentation_0/ --name n --type cnn_vae --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_vae_starting_feature_map 4 --cnn_vae_growth_factor 2 --cnn_vae_kernel_size 3
source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/cnn_vae/start_feature_maps_4/growth_factor_3/data_augmentation_0/ --name n --type cnn_vae --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_vae_starting_feature_map 4 --cnn_vae_growth_factor 3 --cnn_vae_kernel_size 3
source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/cnn_vae/start_feature_maps_8/growth_factor_2/data_augmentation_0/ --name n --type cnn_vae --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_vae_starting_feature_map 8 --cnn_vae_growth_factor 2 --cnn_vae_kernel_size 3
source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/cnn_vae/start_feature_maps_8/growth_factor_3/data_augmentation_0/ --name n --type cnn_vae --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_vae_starting_feature_map 8 --cnn_vae_growth_factor 3 --cnn_vae_kernel_size 3
source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/cnn_vae/start_feature_maps_2/growth_factor_2/data_augmentation_1/ --name n --type cnn_vae --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_vae_starting_feature_map 2 --cnn_vae_growth_factor 2 --cnn_vae_kernel_size 3
source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/cnn_vae/start_feature_maps_2/growth_factor_3/data_augmentation_1/ --name n --type cnn_vae --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_vae_starting_feature_map 2 --cnn_vae_growth_factor 3 --cnn_vae_kernel_size 3
source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/cnn_vae/start_feature_maps_4/growth_factor_2/data_augmentation_1/ --name n --type cnn_vae --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_vae_starting_feature_map 4 --cnn_vae_growth_factor 2 --cnn_vae_kernel_size 3
source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/cnn_vae/start_feature_maps_4/growth_factor_3/data_augmentation_1/ --name n --type cnn_vae --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_vae_starting_feature_map 4 --cnn_vae_growth_factor 3 --cnn_vae_kernel_size 3
source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/cnn_vae/start_feature_maps_8/growth_factor_2/data_augmentation_1/ --name n --type cnn_vae --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_vae_starting_feature_map 8 --cnn_vae_growth_factor 2 --cnn_vae_kernel_size 3
source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/cnn_vae/start_feature_maps_8/growth_factor_3/data_augmentation_1/ --name n --type cnn_vae --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_vae_starting_feature_map 8 --cnn_vae_growth_factor 3 --cnn_vae_kernel_size 3


source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/small_cnn/first_output_channels_2/growth_rate_2/data_augmentation_0/ --name n --type small_cnn --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --small_cnn_kernel_size 3 --small_cnn_first_output_channels 2 --small_cnn_growth_rate 2
source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/small_cnn/first_output_channels_2/growth_rate_3/data_augmentation_0/ --name n --type small_cnn --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --small_cnn_kernel_size 3 --small_cnn_first_output_channels 2 --small_cnn_growth_rate 3
source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/small_cnn/first_output_channels_4/growth_rate_2/data_augmentation_0/ --name n --type small_cnn --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --small_cnn_kernel_size 3 --small_cnn_first_output_channels 4 --small_cnn_growth_rate 2
source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/small_cnn/first_output_channels_4/growth_rate_3/data_augmentation_0/ --name n --type small_cnn --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --small_cnn_kernel_size 3 --small_cnn_first_output_channels 4 --small_cnn_growth_rate 3
source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/small_cnn/first_output_channels_8/growth_rate_2/data_augmentation_0/ --name n --type small_cnn --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --small_cnn_kernel_size 3 --small_cnn_first_output_channels 8 --small_cnn_growth_rate 2
source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/small_cnn/first_output_channels_8/growth_rate_3/data_augmentation_0/ --name n --type small_cnn --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --small_cnn_kernel_size 3 --small_cnn_first_output_channels 8 --small_cnn_growth_rate 3
source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/small_cnn/first_output_channels_2/growth_rate_2/data_augmentation_1/ --name n --type small_cnn --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --small_cnn_kernel_size 3 --small_cnn_first_output_channels 2 --small_cnn_growth_rate 2
source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/small_cnn/first_output_channels_2/growth_rate_3/data_augmentation_1/ --name n --type small_cnn --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --small_cnn_kernel_size 3 --small_cnn_first_output_channels 2 --small_cnn_growth_rate 3
source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/small_cnn/first_output_channels_4/growth_rate_2/data_augmentation_1/ --name n --type small_cnn --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --small_cnn_kernel_size 3 --small_cnn_first_output_channels 4 --small_cnn_growth_rate 2
source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/small_cnn/first_output_channels_4/growth_rate_3/data_augmentation_1/ --name n --type small_cnn --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --small_cnn_kernel_size 3 --small_cnn_first_output_channels 4 --small_cnn_growth_rate 3
source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/small_cnn/first_output_channels_8/growth_rate_2/data_augmentation_1/ --name n --type small_cnn --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --small_cnn_kernel_size 3 --small_cnn_first_output_channels 8 --small_cnn_growth_rate 2
source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/small_cnn/first_output_channels_8/growth_rate_3/data_augmentation_1/ --name n --type small_cnn --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --small_cnn_kernel_size 3 --small_cnn_first_output_channels 8 --small_cnn_growth_rate 3


source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/small_cnn_dense_blocks/growth_rate_2/data_augmentation_0/ --name n --type small_cnn_dense_blocks --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --small_cnn_dense_blocks_kernel_size 3 --small_cnn_dense_blocks_growth_rate 2
source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/small_cnn_dense_blocks/growth_rate_4/data_augmentation_0/ --name n --type small_cnn_dense_blocks --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --small_cnn_dense_blocks_kernel_size 3 --small_cnn_dense_blocks_growth_rate 4
source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/small_cnn_dense_blocks/growth_rate_8/data_augmentation_0/ --name n --type small_cnn_dense_blocks --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --small_cnn_dense_blocks_kernel_size 3 --small_cnn_dense_blocks_growth_rate 8
source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/small_cnn_dense_blocks/growth_rate_16/data_augmentation_0/ --name n --type small_cnn_dense_blocks --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --small_cnn_dense_blocks_kernel_size 3 --small_cnn_dense_blocks_growth_rate 16
source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/small_cnn_dense_blocks/growth_rate_32/data_augmentation_0/ --name n --type small_cnn_dense_blocks --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --small_cnn_dense_blocks_kernel_size 3 --small_cnn_dense_blocks_growth_rate 32
source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/small_cnn_dense_blocks/growth_rate_2/data_augmentation_1/ --name n --type small_cnn_dense_blocks --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --small_cnn_dense_blocks_kernel_size 3 --small_cnn_dense_blocks_growth_rate 2
source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/small_cnn_dense_blocks/growth_rate_4/data_augmentation_1/ --name n --type small_cnn_dense_blocks --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --small_cnn_dense_blocks_kernel_size 3 --small_cnn_dense_blocks_growth_rate 4
source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/small_cnn_dense_blocks/growth_rate_8/data_augmentation_1/ --name n --type small_cnn_dense_blocks --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --small_cnn_dense_blocks_kernel_size 3 --small_cnn_dense_blocks_growth_rate 8
source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/small_cnn_dense_blocks/growth_rate_16/data_augmentation_1/ --name n --type small_cnn_dense_blocks --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --small_cnn_dense_blocks_kernel_size 3 --small_cnn_dense_blocks_growth_rate 16
source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/small_cnn_dense_blocks/growth_rate_32/data_augmentation_1/ --name n --type small_cnn_dense_blocks --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --small_cnn_dense_blocks_kernel_size 3 --small_cnn_dense_blocks_growth_rate 32


source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/vgg16_backend_autoencoder/train_back_end_0/data_augmentation_0/ --name n --type vgg16_backend_autoencoder --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0
source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/vgg16_backend_autoencoder/train_back_end_1/data_augmentation_0/ --name n --type vgg16_backend_autoencoder --batch_size 20 --vgg16_backend_autoencoder_train_backend --learning_rate 0.001 --epoch_count 1 --weight_decay 0
source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/vgg16_backend_autoencoder/train_back_end_0/data_augmentation_1/ --name n --type vgg16_backend_autoencoder --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0
source train_tunnel.sh --output_path ~/CuriousNetwork_out/tunnel/vgg16_backend_autoencoder/train_back_end_1/data_augmentation_1/ --name n --type vgg16_backend_autoencoder --batch_size 20 --data_augmentation --vgg16_backend_autoencoder_train_backend --learning_rate 0.001 --epoch_count 1 --weight_decay 0


source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/cnn_autoencoder/start_feature_maps_2/growth_factor_2/data_augmentation_0/ --name n --type cnn_autoencoder --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_autoencoder_starting_feature_map 2 --cnn_autoencoder_growth_factor 2 --cnn_autoencoder_kernel_size 3
source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/cnn_autoencoder/start_feature_maps_2/growth_factor_3/data_augmentation_0/ --name n --type cnn_autoencoder --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_autoencoder_starting_feature_map 2 --cnn_autoencoder_growth_factor 3 --cnn_autoencoder_kernel_size 3
source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/cnn_autoencoder/start_feature_maps_4/growth_factor_2/data_augmentation_0/ --name n --type cnn_autoencoder --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_autoencoder_starting_feature_map 4 --cnn_autoencoder_growth_factor 2 --cnn_autoencoder_kernel_size 3
source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/cnn_autoencoder/start_feature_maps_4/growth_factor_3/data_augmentation_0/ --name n --type cnn_autoencoder --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_autoencoder_starting_feature_map 4 --cnn_autoencoder_growth_factor 3 --cnn_autoencoder_kernel_size 3
source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/cnn_autoencoder/start_feature_maps_8/growth_factor_2/data_augmentation_0/ --name n --type cnn_autoencoder --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_autoencoder_starting_feature_map 8 --cnn_autoencoder_growth_factor 2 --cnn_autoencoder_kernel_size 3
source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/cnn_autoencoder/start_feature_maps_8/growth_factor_3/data_augmentation_0/ --name n --type cnn_autoencoder --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_autoencoder_starting_feature_map 8 --cnn_autoencoder_growth_factor 3 --cnn_autoencoder_kernel_size 3
source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/cnn_autoencoder/start_feature_maps_2/growth_factor_2/data_augmentation_1/ --name n --type cnn_autoencoder --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_autoencoder_starting_feature_map 2 --cnn_autoencoder_growth_factor 2 --cnn_autoencoder_kernel_size 3
source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/cnn_autoencoder/start_feature_maps_2/growth_factor_3/data_augmentation_1/ --name n --type cnn_autoencoder --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_autoencoder_starting_feature_map 2 --cnn_autoencoder_growth_factor 3 --cnn_autoencoder_kernel_size 3
source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/cnn_autoencoder/start_feature_maps_4/growth_factor_2/data_augmentation_1/ --name n --type cnn_autoencoder --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_autoencoder_starting_feature_map 4 --cnn_autoencoder_growth_factor 2 --cnn_autoencoder_kernel_size 3
source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/cnn_autoencoder/start_feature_maps_4/growth_factor_3/data_augmentation_1/ --name n --type cnn_autoencoder --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_autoencoder_starting_feature_map 4 --cnn_autoencoder_growth_factor 3 --cnn_autoencoder_kernel_size 3
source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/cnn_autoencoder/start_feature_maps_8/growth_factor_2/data_augmentation_1/ --name n --type cnn_autoencoder --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_autoencoder_starting_feature_map 8 --cnn_autoencoder_growth_factor 2 --cnn_autoencoder_kernel_size 3
source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/cnn_autoencoder/start_feature_maps_8/growth_factor_3/data_augmentation_1/ --name n --type cnn_autoencoder --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_autoencoder_starting_feature_map 8 --cnn_autoencoder_growth_factor 3 --cnn_autoencoder_kernel_size 3


source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/cnn_vae/start_feature_maps_2/growth_factor_2/data_augmentation_0/ --name n --type cnn_vae --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_vae_starting_feature_map 2 --cnn_vae_growth_factor 2 --cnn_vae_kernel_size 3
source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/cnn_vae/start_feature_maps_2/growth_factor_3/data_augmentation_0/ --name n --type cnn_vae --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_vae_starting_feature_map 2 --cnn_vae_growth_factor 3 --cnn_vae_kernel_size 3
source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/cnn_vae/start_feature_maps_4/growth_factor_2/data_augmentation_0/ --name n --type cnn_vae --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_vae_starting_feature_map 4 --cnn_vae_growth_factor 2 --cnn_vae_kernel_size 3
source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/cnn_vae/start_feature_maps_4/growth_factor_3/data_augmentation_0/ --name n --type cnn_vae --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_vae_starting_feature_map 4 --cnn_vae_growth_factor 3 --cnn_vae_kernel_size 3
source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/cnn_vae/start_feature_maps_8/growth_factor_2/data_augmentation_0/ --name n --type cnn_vae --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_vae_starting_feature_map 8 --cnn_vae_growth_factor 2 --cnn_vae_kernel_size 3
source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/cnn_vae/start_feature_maps_8/growth_factor_3/data_augmentation_0/ --name n --type cnn_vae --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_vae_starting_feature_map 8 --cnn_vae_growth_factor 3 --cnn_vae_kernel_size 3
source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/cnn_vae/start_feature_maps_2/growth_factor_2/data_augmentation_1/ --name n --type cnn_vae --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_vae_starting_feature_map 2 --cnn_vae_growth_factor 2 --cnn_vae_kernel_size 3
source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/cnn_vae/start_feature_maps_2/growth_factor_3/data_augmentation_1/ --name n --type cnn_vae --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_vae_starting_feature_map 2 --cnn_vae_growth_factor 3 --cnn_vae_kernel_size 3
source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/cnn_vae/start_feature_maps_4/growth_factor_2/data_augmentation_1/ --name n --type cnn_vae --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_vae_starting_feature_map 4 --cnn_vae_growth_factor 2 --cnn_vae_kernel_size 3
source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/cnn_vae/start_feature_maps_4/growth_factor_3/data_augmentation_1/ --name n --type cnn_vae --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_vae_starting_feature_map 4 --cnn_vae_growth_factor 3 --cnn_vae_kernel_size 3
source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/cnn_vae/start_feature_maps_8/growth_factor_2/data_augmentation_1/ --name n --type cnn_vae --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_vae_starting_feature_map 8 --cnn_vae_growth_factor 2 --cnn_vae_kernel_size 3
source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/cnn_vae/start_feature_maps_8/growth_factor_3/data_augmentation_1/ --name n --type cnn_vae --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --cnn_vae_starting_feature_map 8 --cnn_vae_growth_factor 3 --cnn_vae_kernel_size 3


source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/small_cnn/first_output_channels_2/growth_rate_2/data_augmentation_0/ --name n --type small_cnn --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --small_cnn_kernel_size 3 --small_cnn_first_output_channels 2 --small_cnn_growth_rate 2
source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/small_cnn/first_output_channels_2/growth_rate_3/data_augmentation_0/ --name n --type small_cnn --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --small_cnn_kernel_size 3 --small_cnn_first_output_channels 2 --small_cnn_growth_rate 3
source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/small_cnn/first_output_channels_4/growth_rate_2/data_augmentation_0/ --name n --type small_cnn --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --small_cnn_kernel_size 3 --small_cnn_first_output_channels 4 --small_cnn_growth_rate 2
source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/small_cnn/first_output_channels_4/growth_rate_3/data_augmentation_0/ --name n --type small_cnn --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --small_cnn_kernel_size 3 --small_cnn_first_output_channels 4 --small_cnn_growth_rate 3
source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/small_cnn/first_output_channels_8/growth_rate_2/data_augmentation_0/ --name n --type small_cnn --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --small_cnn_kernel_size 3 --small_cnn_first_output_channels 8 --small_cnn_growth_rate 2
source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/small_cnn/first_output_channels_8/growth_rate_3/data_augmentation_0/ --name n --type small_cnn --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --small_cnn_kernel_size 3 --small_cnn_first_output_channels 8 --small_cnn_growth_rate 3
source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/small_cnn/first_output_channels_2/growth_rate_2/data_augmentation_1/ --name n --type small_cnn --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --small_cnn_kernel_size 3 --small_cnn_first_output_channels 2 --small_cnn_growth_rate 2
source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/small_cnn/first_output_channels_2/growth_rate_3/data_augmentation_1/ --name n --type small_cnn --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --small_cnn_kernel_size 3 --small_cnn_first_output_channels 2 --small_cnn_growth_rate 3
source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/small_cnn/first_output_channels_4/growth_rate_2/data_augmentation_1/ --name n --type small_cnn --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --small_cnn_kernel_size 3 --small_cnn_first_output_channels 4 --small_cnn_growth_rate 2
source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/small_cnn/first_output_channels_4/growth_rate_3/data_augmentation_1/ --name n --type small_cnn --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --small_cnn_kernel_size 3 --small_cnn_first_output_channels 4 --small_cnn_growth_rate 3
source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/small_cnn/first_output_channels_8/growth_rate_2/data_augmentation_1/ --name n --type small_cnn --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --small_cnn_kernel_size 3 --small_cnn_first_output_channels 8 --small_cnn_growth_rate 2
source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/small_cnn/first_output_channels_8/growth_rate_3/data_augmentation_1/ --name n --type small_cnn --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --small_cnn_kernel_size 3 --small_cnn_first_output_channels 8 --small_cnn_growth_rate 3


source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/small_cnn_dense_blocks/growth_rate_2/data_augmentation_0/ --name n --type small_cnn_dense_blocks --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --small_cnn_dense_blocks_kernel_size 3 --small_cnn_dense_blocks_growth_rate 2
source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/small_cnn_dense_blocks/growth_rate_4/data_augmentation_0/ --name n --type small_cnn_dense_blocks --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --small_cnn_dense_blocks_kernel_size 3 --small_cnn_dense_blocks_growth_rate 4
source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/small_cnn_dense_blocks/growth_rate_8/data_augmentation_0/ --name n --type small_cnn_dense_blocks --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --small_cnn_dense_blocks_kernel_size 3 --small_cnn_dense_blocks_growth_rate 8
source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/small_cnn_dense_blocks/growth_rate_16/data_augmentation_0/ --name n --type small_cnn_dense_blocks --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --small_cnn_dense_blocks_kernel_size 3 --small_cnn_dense_blocks_growth_rate 16
source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/small_cnn_dense_blocks/growth_rate_32/data_augmentation_0/ --name n --type small_cnn_dense_blocks --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --small_cnn_dense_blocks_kernel_size 3 --small_cnn_dense_blocks_growth_rate 32
source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/small_cnn_dense_blocks/growth_rate_2/data_augmentation_1/ --name n --type small_cnn_dense_blocks --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --small_cnn_dense_blocks_kernel_size 3 --small_cnn_dense_blocks_growth_rate 2
source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/small_cnn_dense_blocks/growth_rate_4/data_augmentation_1/ --name n --type small_cnn_dense_blocks --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --small_cnn_dense_blocks_kernel_size 3 --small_cnn_dense_blocks_growth_rate 4
source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/small_cnn_dense_blocks/growth_rate_8/data_augmentation_1/ --name n --type small_cnn_dense_blocks --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --small_cnn_dense_blocks_kernel_size 3 --small_cnn_dense_blocks_growth_rate 8
source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/small_cnn_dense_blocks/growth_rate_16/data_augmentation_1/ --name n --type small_cnn_dense_blocks --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --small_cnn_dense_blocks_kernel_size 3 --small_cnn_dense_blocks_growth_rate 16
source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/small_cnn_dense_blocks/growth_rate_32/data_augmentation_1/ --name n --type small_cnn_dense_blocks --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0 --small_cnn_dense_blocks_kernel_size 3 --small_cnn_dense_blocks_growth_rate 32


source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/vgg16_backend_autoencoder/train_back_end_0/data_augmentation_0/ --name n --type vgg16_backend_autoencoder --batch_size 20 --learning_rate 0.001 --epoch_count 1 --weight_decay 0
source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/vgg16_backend_autoencoder/train_back_end_1/data_augmentation_0/ --name n --type vgg16_backend_autoencoder --batch_size 20 --vgg16_backend_autoencoder_train_backend --learning_rate 0.001 --epoch_count 1 --weight_decay 0
source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/vgg16_backend_autoencoder/train_back_end_0/data_augmentation_1/ --name n --type vgg16_backend_autoencoder --batch_size 20 --data_augmentation --learning_rate 0.001 --epoch_count 1 --weight_decay 0
source train_corridor.sh --output_path ~/CuriousNetwork_out/corridor/vgg16_backend_autoencoder/train_back_end_1/data_augmentation_1/ --name n --type vgg16_backend_autoencoder --batch_size 20 --data_augmentation --vgg16_backend_autoencoder_train_backend --learning_rate 0.001 --epoch_count 1 --weight_decay 0
