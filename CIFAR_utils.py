import torchvision
import torch
import sklearn.metrics as metrics
from loss_functions import IntermediateLoss
import pdb


def CIFAR_test(model, opt):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))])

    if opt.dataset == 'CIFAR10':
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform)
    elif opt.dataset == 'CIFAR100':
        testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.bs_ft,
                                            shuffle=False, num_workers=4)

    gt = list()
    pred = list()
    model = model.to(opt.device)
    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(opt.device)
            labels = labels.to(opt.device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            gt.append(labels)
            pred.append(predicted)

    gt = torch.cat(gt).cpu().detach().numpy()
    pred = torch.cat(pred).cpu().detach().numpy()
    accuracy = metrics.accuracy_score(gt, pred)
    conf_matrix = metrics.confusion_matrix(gt, pred)
    acc_per_class = conf_matrix.diagonal()/conf_matrix.sum(axis=1)

    return accuracy * 100, acc_per_class * 100


def CIFAR_train(opt, optimizer, gt_labels, backbone, pruned_backbone, loss_func, loss_mean):
    # Get dataloader
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])])
    
    if opt.dataset == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                                download=True, transform=transform)
    elif opt.dataset == 'CIFAR100':
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, 
                                                download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.bs_ft,
                                            shuffle=True, num_workers=4)

    # Set the models to the right mode
    backbone.eval()
    pruned_backbone.train()

    # Train loop for each epoch
    for iter, data in enumerate(trainloader):
        optimizer.zero_grad()

        inputs, labels = data
        inputs = inputs.to(opt.device)
        # labels = labels.to(opt.device)

        if isinstance(loss_func, IntermediateLoss):
            loss_func.clear_hooks()

        gt_preds = backbone(inputs).detach()
        preds = pruned_backbone(inputs)

        # append gt labels for evaluation
        gt_labels.append(labels)

        # Calculate loss
        if isinstance(loss_func, IntermediateLoss):
            p_loss_mse = loss_func.get_intermediate_loss()
        else:
            p_loss_mse = loss_func(preds, gt_preds)

        p_loss_mse.backward()
        optimizer.step()

        loss_mean = (loss_mean * (iter) + p_loss_mse.detach().cpu()) / (iter + 1)

    return pruned_backbone, gt_labels, loss_mean