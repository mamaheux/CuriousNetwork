#!bin/bash

#python3 /home/ggs22/repos/CuriousNetwork/nn/run.py -i /home/ggs22/repos/CuriousNetwork/dataset/jpg/corridor_test -o /home/ggs22/repos/CuriousNetwork/run_output/corridor/ -t vgg16_backend_autoencoder -w ~/Documents/USherbrooke/IFT725/Projet/results/corridor/data_augmentation_0/n_epoch_9.pth --threshold 2.5
#python3 /home/ggs22/repos/CuriousNetwork/nn/run.py -i /home/ggs22/repos/CuriousNetwork/dataset/jpg/tunnel_test -o /home/ggs22/repos/CuriousNetwork/run_output/tunnel/ -t vgg16_backend_autoencoder -w ~/Documents/USherbrooke/IFT725/Projet/results/corridor/data_augmentation_0/n_epoch_9.pth --threshold 2.5
python3 /home/ggs22/repos/CuriousNetwork/nn/run.py -i /home/ggs22/repos/CuriousNetwork/dataset/jpg/corridor_test -o /home/ggs22/repos/CuriousNetwork/run_output/tr_tunnel_tst_corridor/ -t vgg16_backend_autoencoder -w ~/Documents/USherbrooke/IFT725/Projet/results/tunnel/data_augmentation_0/n_epoch_9.pth --threshold 2.5
python3 /home/ggs22/repos/CuriousNetwork/nn/run.py -i /home/ggs22/repos/CuriousNetwork/dataset/jpg/tunnel_test -o /home/ggs22/repos/CuriousNetwork/run_output/tr_corridor_tst_tunnel/ -t vgg16_backend_autoencoder -w ~/Documents/USherbrooke/IFT725/Projet/results/corridor/data_augmentation_0/n_epoch_9.pth --threshold 2.5

