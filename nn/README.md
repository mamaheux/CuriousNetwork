# Neural Networks
These are the models implemented and tested for the Curious Network project. They can be trained then
called from run.py in order to test the "curiosity singal" (i.e. the error on the re-encoded annotated pictures).

## Create a virtual environment
```bash
python3 -m venv env
source env/bin/activate
```

## Dependencies installation
```bash
source env/bin/activate
pip install -r requirements.txt
```

## Launch the training
```bash
source env/bin/activate
python train.py    -h, --help            show this help message and exit
                          --use_gpu
                          --train_path TRAIN_PATH
                                                Choose the training database path
                          --val_path VAL_PATH   Choose the validation database path
                          --test_path TEST_PATH
                                                Choose the validation database path
                          -o OUTPUT_PATH, --output_path OUTPUT_PATH
                                                Choose the output path
                          -n NAME, --name NAME  Choose the model name
                          -t {cnn_autoencoder,cnn_vae,vgg16_backend_autoencoder,small_cnn,small_cnn_dense_blocks}, --type {cnn_autoencoder,cnn_vae,vgg16_backend_autoencoder,small_cnn,small_cnn_dense_blocks}
                                                Choose the network type
                          -s BATCH_SIZE, --batch_size BATCH_SIZE
                                                Set the batch size for the training
                          -d, --data_augmentation
                                                Use data augmentation or not
                          --learning_rate LEARNING_RATE
                                                Choose the learning rate
                          --epoch_count EPOCH_COUNT
                                                Choose the epoch count
                          --weight_decay WEIGHT_DECAY
                                                Choose the weight decay
                          --cnn_autoencoder_starting_feature_map CNN_AUTOENCODER_STARTING_FEATURE_MAP
                                                Choose the number of starting feature maps for the
                                                autoencoder
                          --cnn_autoencoder_growth_factor CNN_AUTOENCODER_GROWTH_FACTOR
                                                Choose the basis of the coefficient by which the
                                                featuremaps are goind to be multiplied from a layer to
                                                the next
                          --cnn_autoencoder_kernel_size CNN_AUTOENCODER_KERNEL_SIZE
                                                Choose the starting kernel size
                          --cnn_vae_starting_feature_map CNN_VAE_STARTING_FEATURE_MAP
                                                Choose the number of starting feature maps for the
                                                autoencoder
                          --cnn_vae_growth_factor CNN_VAE_GROWTH_FACTOR
                                                Choose the basis of the coefficient by which the
                                                featuremaps are goind to be multiplied from a layer to
                                                the next
                          --cnn_vae_kernel_size CNN_VAE_KERNEL_SIZE
                                                Choose the starting kernel size
                          --vgg16_backend_autoencoder_train_backend
                                                Train the backend
                          --small_cnn_kernel_size SMALL_CNN_KERNEL_SIZE
                                                Set the value for small_cnn
                          --small_cnn_first_output_channels SMALL_CNN_FIRST_OUTPUT_CHANNELS
                                                Set the value for small_cnn
                          --small_cnn_growth_rate SMALL_CNN_GROWTH_RATE
                                                Set the value for small_cnn
                          --small_cnn_dense_blocks_kernel_size SMALL_CNN_DENSE_BLOCKS_KERNEL_SIZE
                                                Set the value for small_cnn_dense_blocks
                          --small_cnn_dense_blocks_growth_rate SMALL_CNN_DENSE_BLOCKS_GROWTH_RATE
                                                Set the value for small_cnn_dense_blocks