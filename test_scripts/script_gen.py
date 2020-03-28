# Common parameters
use_gpu = True
train_path = ''
val_path = ''
test_path = ''
output_path = '~/CuriousNetwork_out'
model_name = ''

model_types = ['cnn_autoencoder', 'cnn_vae', 'small_cnn', 'small_cnn_dense_blocks']  # 'vgg16_backend_autoencoder'
datasets = ["tunnel", "corridor"]

batch_size = 20
data_augmentation = [0, 1]
learning_rate = 0.001
epoch_count = 20
weight_decay = 0

# CNN autoencoder & CNN vanilla
starting_feature_map = [2, 4, 8]
growth_factor = [2, 3]
kernel_size = 3

# VGG16 backend
train_back_end = [0, 1]

# CNN autoencoder & CNN vanilla
small_cnn_starting_feature_map = [2, 4, 8, 16]
small_cnn_growth_factor = [2, 4]
small_cnn_kernel_size = 3

# small cnn dense blocks
dense_blocks_growth_rates = [2, 4, 8, 16, 32]

for dataset in datasets:
    for model_type in model_types:
        if model_type == 'cnn_autoencoder':
            for da in data_augmentation:
                for fm in starting_feature_map:
                    for gf in growth_factor:
                        call_str = f'sbatch train_{dataset}.sh ' \
                                   f'--use_gpu ' \
                                   f'--output_path ' \
                                       f'{output_path}/{dataset}/{model_type}/start_feature_maps_{fm}/' \
                                       f'growth_factor_{gf}/ '\
                                   f'--name n --type {model_type} '\
                                   f'--batch_size {batch_size} ' +\
                                   f'--data_augmentation' * da +\
                                   f'--learning_rate {learning_rate} '\
                                   f'--epoch_count {epoch_count} '\
                                   f'--weight_decay {weight_decay} '\
                                   f'--cnn_autoencoder_starting_feature_map {fm} '\
                                   f'--cnn_autoencoder_growth_factor {gf} '\
                                   f'--cnn_autoencoder_kernel_size {3}'
                        print(call_str)
        if model_type == 'cnn_vae':
            for da in data_augmentation:
                for fm in starting_feature_map:
                    for gf in growth_factor:
                        call_str = f'sbatch train_{dataset}.sh ' \
                                   f'--use_gpu ' \
                                   f'--output_path ' \
                                       f'{output_path}/{dataset}/{model_type}/start_feature_maps_{fm}/' \
                                       f'growth_factor_{gf}/ ' \
                                   f'--name n --type {model_type} ' \
                                   f'--batch_size {batch_size} ' +\
                                   f'--data_augmentation ' * da +\
                                   f'--learning_rate {learning_rate} ' \
                                   f'--epoch_count {epoch_count} ' \
                                   f'--weight_decay {weight_decay} ' \
                                   f'--cnn_autoencoder_starting_feature_map {fm} ' \
                                   f'--cnn_autoencoder_growth_factor {gf} ' \
                                   f'--cnn_autoencoder_kernel_size {3}'
                        print(call_str)
        if model_type == 'vgg16_backend_autoencoder':
            for da in data_augmentation:
                for fm in starting_feature_map:
                    for tb in train_back_end:
                        call_str = f'sbatch train_{dataset}.sh ' \
                                   f'--use_gpu ' \
                                   f'--output_path ' \
                                       f'{output_path}/{dataset}/{model_type}/' \
                                       f'data_augmentation_{da}/ ' \
                                   f'--name n ' \
                                   f'--type {model_type} ' \
                                   f'--batch_size {batch_size} ' +\
                                   f'--data_augmentation' * da +\
                                   f'--vgg16_backend_autoencoder_train_backend' * tb +\
                                   f'--learning_rate {learning_rate} ' \
                                   f'--epoch_count {epoch_count} ' \
                                   f'--weight_decay {weight_decay}'
                        print(call_str)
        if model_type == 'small_cnn':
            for da in data_augmentation:
                for fm in small_cnn_starting_feature_map:
                    for gf in growth_factor:
                        call_str = f'sbatch train_{dataset}.sh ' \
                                   f'--use_gpu ' \
                                   f'--output_path ' \
                                       f'{output_path}/{dataset}/{model_type}/learning_rate_{learning_rate}/' \
                                       f'first_output_channels_{fm}/growth_rate_{gf}/ ' \
                                   f'--name n ' \
                                   f'--type {model_type} ' \
                                   f'--batch_size {batch_size} ' +\
                                   f'--data_augmentation' * da +\
                                   f'--learning_rate {learning_rate} ' \
                                   f'--epoch_count {epoch_count} ' \
                                   f'--weight_decay {weight_decay} ' \
                                   f'--small_cnn_kernel_size {3} ' \
                                   f'--small_cnn_first_output_channels {fm} ' \
                                   f'--small_cnn_growth_rate {gf}'
                        print(call_str)
        if model_type == 'small_cnn_dense_blocks':
            for da in data_augmentation:
                for gr in dense_blocks_growth_rates:
                    call_str = f'sbatch train_{dataset}.sh ' \
                               f'--use_gpu ' \
                               f'--output_path ' \
                               f'{output_path}/{dataset}/{model_type}/learning_rate_{learning_rate}/growth_rate_{gr}/ ' \
                               f'--name n ' \
                               f'--type {model_type} ' \
                               f'--batch_size {batch_size} ' + \
                               f'--data_augmentation' * da + \
                               f'--learning_rate {learning_rate} ' \
                               f'--epoch_count {epoch_count} ' \
                               f'--weight_decay {weight_decay} ' \
                               f'--small_cnn_dense_blocks_kernel_size {3} ' \
                               f'--small_cnn_dense_blocks_growth_rate {gr}'
                    print(call_str)
