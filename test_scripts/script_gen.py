# Common parameters
use_gpu = True
train_path = ''
val_path = ''
test_path = ''
output_path = '~/CuriousNetwork_out'
model_name = ''

model_types = ['cnn_autoencoder', 'cnn_vae', 'vgg16_backend_autoencoder', 'small_cnn']
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
train_back_end = True

# CNN autoencoder & CNN vanilla
small_cnn_starting_feature_map = [2, 4, 8, 16]
small_cnn_growth_factor = [2, 4]
small_cnn_kernel_size = 3

'--use_gpu'
'--train_path'
'--output_path'
'--test_path'
'--output_path'
'--name'

'--type'
'--batch_size'
'-data_augmentation'
'--learning_rate'
'--epoch_count'
'--weight_decay'

    # cnn_autoencoder arguments
'--cnn_autoencoder_starting_feature_map'
'--cnn_autoencoder_growth_factor'
'--cnn_autoencoder_kernel_size'

    # cnn_vae arguments
'--cnn_vae_starting_feature_map'
'--cnn_vae_growth_factor'
'--cnn_vae_kernel_size'

'--vgg16_backend_autoencoder_train_backend'

    # small_cnn arguments
'--small_cnn_kernel_size'
'--small_cnn_first_output_channels'
'--small_cnn_growth_rate'


for dataset in ["tunnel", "corridor"]:
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
                    for gf in growth_factor:
                        call_str = f'sbatch train_{dataset}.sh ' \
                                   f'--use_gpu ' \
                                   f'--output_path {output_path}/{dataset}/{model_type}/data_augmentation_{da}/ ' \
                                   f'--name n ' \
                                   f'--type {model_type} ' \
                                   f'--batch_size {batch_size} ' +\
                                   f'--data_augmentation' * da + \
                                   f'--learning_rate {learning_rate} ' \
                                                                                                f'--epoch_count {epoch_count} --weight_decay {weight_decay}'
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
