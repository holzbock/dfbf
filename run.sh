###### sparsity 0.1 ######
# COCO ResNet50
python3 main.py --model resnet50 --dataset COCO --task detection --num_batches_ft 100 --lr_ft 0.01 --bs_ft 16 --sparsity 0.1 --epochs_ft 5 --lr_steps_ft 3 --loss_weight_method 2 --img_size 250 --layer_depending_sparsity True
# COCO VGG16
python3 main.py --model vgg16 --dataset COCO --task detection --num_batches_ft 100 --lr_ft 0.01 --bs_ft 16 --sparsity 0.1 --epochs_ft 65 --lr_steps_ft 60 --loss_weight_method 5 --img_size 250 --layer_depending_sparsity True
# VOC ResNet50
python3 main.py --model resnet50 --dataset VOC --task detection --num_batches_ft 100 --lr_ft 0.01 --bs_ft 16 --sparsity 0.1 --epochs_ft 10 --lr_steps_ft 7 --loss_weight_method 2 --img_size 250 --layer_depending_sparsity True
# COCO OpenPifPaf
python3 main.py --model resnet50 --dataset COCO --task pose --num_batches_ft 100 --lr_ft 0.01 --bs_ft 16 --sparsity 0.1 --epochs_ft 200 --lr_steps_ft 150 --loss_weight_method 8 --img_size 160 --layer_depending_sparsity True

###### sparsity 0.5 ######
# CIFAR10 VGG16
python3 main.py --model vgg16 --dataset CIFAR10 --task classification --num_classes 10 --num_batches_ft 50 --lr_ft 0.01 --sparsity 0.5
# CIFAR10 VGG19
python3 main.py --model vgg19 --dataset CIFAR10 --task classification --num_classes 10 --num_batches_ft 50 --lr_ft 0.01 --sparsity 0.5
# CIFAR10 ResNet18
python3 main.py --model resnet18 --dataset CIFAR10 --task classification --num_classes 10 --num_batches_ft 50 --lr_ft 0.01 --sparsity 0.5
# CIFAR10 ResNet34
python3 main.py --model resnet34 --dataset CIFAR10 --task classification --num_classes 10 --num_batches_ft 50 --lr_ft 0.01 --sparsity 0.5