# -*- coding: utf-8 -*-
# @Author  : Alisa
# @File    : main(pretrain).py
# @Software: PyCharm
import warnings
from evaluate import evaluate, train_test_split_few, context_inference
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.loader import DataLoader
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import torch
import itertools
from torch.optim import Adam
from pargs import pargs
from load_data import load_datasets_with_prompts, TreeDataset, HugeDataset, TreeDataset_PHEME, TreeDataset_UPFD, \
    CovidDataset
from model import BiGCN_graphcl
from augmentation import augment
from torch_geometric import seed_everything

warnings.filterwarnings("ignore")


def get_scheduler(optimizer, use_scheduler=True, epochs=1000):
    if use_scheduler:
        scheduler = lambda epoch: (1 + np.cos(epoch * np.pi / epochs)) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
    else:
        scheduler = None

    return scheduler


def pre_train(loaders, aug1, aug2, model, optimizer, device):
    """
    Pre-train the model with multiple DataLoaders.

    :param loaders: List of DataLoaders for the datasets.
    :param aug1: String specifying the first set of augmentations.
    :param aug2: String specifying the second set of augmentations.
    :param model: The model to train.
    :param optimizer: Optimizer for the training process.
    :param device: Device to perform computations on (e.g., 'cuda' or 'cpu').
    :return: Average loss over all datasets.
    """
    model.train()
    total_loss = 0

    # Split augmentation strategies
    augs1 = aug1.split('||')
    augs2 = aug2.split('||')

    # Iterate through batches from each DataLoader, using itertools.zip_longest to handle different lengths
    for i, batches in enumerate(itertools.zip_longest(*loaders, fillvalue=None)):
        optimizer.zero_grad()

        augmented_data1 = []
        augmented_data2 = []

        # Process each batch from the different loaders
        for idx, batch in enumerate(batches):
            if batch is not None:  # Ensure the batch is not None (handle shorter datasets)
                batch = batch.to(device)
                aug_data1 = augment(batch, augs1)
                aug_data2 = augment(batch, augs2)
                # Attach prompt_key to the batch
                aug_data1.prompt_key = loaders[idx].prompt_key
                aug_data2.prompt_key = loaders[idx].prompt_key
                augmented_data1.append(aug_data1)
                augmented_data2.append(aug_data2)

        # Model forward pass
        out1 = model(*augmented_data1)
        out2 = model(*augmented_data2)
        #############################################
        # augmented_data1 = []
        #
        # # Process each batch from the different loaders
        # for idx, batch in enumerate(batches):
        #     if batch is not None:  # Ensure the batch is not None (handle shorter datasets)
        #         batch = batch.to(device)
        #         aug_data1 = augment(batch, augs1)
        #         augmented_data1.append(aug_data1)
        # # Model forward pass
        # loss = model(*augmented_data1)

        # Compute the loss using the contrastive loss function
        loss = model.loss_graphcl(out1, out2)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    total_loss /= len(loaders)
    return total_loss


def pre_trains(loaders, aug1, aug2, model, optimizer, device):
    """
    Pre-train the model with multiple DataLoaders.

    :param loaders: List of DataLoaders for the datasets.
    :param aug1: String specifying the first set of augmentations.
    :param aug2: String specifying the second set of augmentations.
    :param model: The model to train.
    :param optimizer: Optimizer for the training process.
    :param device: Device to perform computations on (e.g., 'cuda' or 'cpu').
    :return: Average loss over all datasets.
    """
    model.train()
    total_loss = 0

    # Split augmentation strategies
    augs1 = aug1.split('||')
    augs2 = aug2.split('||')

    # Iterate through batches from each DataLoader, using itertools.zip_longest to handle different lengths
    for data in loaders:
        optimizer.zero_grad()
        data = data.to(device)

        aug_data1 = augment(data, augs1)
        aug_data2 = augment(data, augs2)

        out1 = model(aug_data1)
        out2 = model(aug_data2)
        loss = model.loss_graphcl(out1, out2)
        print(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    total_loss /= len(loaders)
    return total_loss

def get_num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_millions(num):
    return round(num / 1e6, 2)


if __name__ == '__main__':

    f1_macros_5 = []

    args = pargs()
    seed_everything(0)
    dataset = args.dataset
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    batch_size = args.batch_size

    weight_decay = args.weight_decay
    epochs = args.epochs

    # Initialize datasets
    # data = TreeDataset("./Data/DRWeiboV3/")
    # data = TreeDataset("../ACL/Data/Weibo/")
    # data = TreeDataset("./Data/Twitter15-tfidf/")
    # data = TreeDataset("./Data/Twitter16-tfidf/")
    # data = TreeDataset_PHEME("./Data/pheme/")
    # data = TreeDataset_UPFD("./Data/politifact/")
    # data = TreeDataset_UPFD("./Data/gossipcop/")
    # data = CovidDataset("./Data/Twitter-COVID19/Twittergraph")
    # data = CovidDataset("./Data/Weibo-COVID19/Weibograph")
    # data = HugeDataset("./Data/Tree/")
    # train_loaders = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=48)
    # target_loader = DataLoader(data, batch_size=args.batch_size, shuffle=False)

    train_loaders, target_loader = load_datasets_with_prompts(args)

    # Model and optimizer initialization
    t = 0.5
    u = 0.5
    model = BiGCN_graphcl(768, args.out_feat, t, u).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    # scheduler = get_scheduler(optimizer, epochs=epochs)
    num_params = to_millions(get_num_params(model))
    print(num_params)
    for epoch in range(1, epochs + 1):
        start_time = time.time()  # ===== 开始计时 =====
        pretrain_loss = pre_train(train_loaders,
                                  args.aug1, args.aug2, model, optimizer, device)
        # scheduler.step()
        end_time = time.time()  # ===== 结束计时 =====
        epoch_time = end_time - start_time
        print(f"Epoch: {epoch}, loss: {pretrain_loss}, time: {epoch_time:.3f} sec")
    # torch.save(model.state_dict(), f"./{dataset}_RAGCL.pt")
    print(dataset)

    # model.load_state_dict(torch.load(f"./{dataset}_RAGCL.pt", map_location=device))

    # Evaluation
    model.eval()
    x_list = []
    y_list = []
    for data in target_loader:
        data = data.to(device)
        embeds = model.get_embeds(data).detach()
        y_list.append(data.y)
        x_list.append(embeds)
    x = torch.cat(x_list, dim=0)
    y = torch.cat(y_list, dim=0)

    ################################################################################################
    for r in [1, 5]:
        mask = train_test_split_few(y.cpu().numpy(), seed=0,
                                    train_examples_per_class=r,
                                    val_size=500, test_size=None)
        train_mask_l = f"{r}_train_mask"
        train_mask = mask['train'].astype(bool)
        val_mask_l = f"{r}_val_mask"
        val_mask = mask['val'].astype(bool)

        test_mask_l = f"{r}_test_mask"
        test_mask = mask['test'].astype(bool)

        evaluate(x, y, device, 0.01, 0.0, train_mask, val_mask, test_mask)
    ############################################################################################
    # for i in range(10):
    #     for r in [1, 5]:
    #         mask = train_test_split_few(y.cpu().numpy(), seed=i,
    #                                     train_examples_per_class=r,
    #                                     val_size=500, test_size=None)
    #         train_mask_l = f"{r}_train_mask"
    #         train_mask = mask['train'].astype(bool)
    #         val_mask_l = f"{r}_val_mask"
    #         val_mask = mask['val'].astype(bool)
    #
    #         test_mask_l = f"{r}_test_mask"
    #         test_mask = mask['test'].astype(bool)
    #
    #         # evaluate(x, y, device, 0.01, 0.0, train_mask, val_mask, test_mask)
    #         f1_macro, f1_micro = context_inference(x, y, device, train_mask, test_mask)
    #         if r == 1:
    #             f1_macros_1.append(f1_macro)
    #         if r == 5:
    #             f1_macros_5.append(f1_macro)
    # # Compute mean and standard deviation
    # f1_macro_1_mean, f1_macro_1_std = np.mean(f1_macros_1), np.std(f1_macros_1)
    #
    # f1_macro_5_mean, f1_macro_5_std = np.mean(f1_macros_5), np.std(f1_macros_5)
    #
    # # Print final results
    # print("\nFinal Results (10 runs) for 1-shot:")
    # print(f"Macro-F1: Mean = {f1_macro_1_mean:.4f}, Std = {f1_macro_1_std:.4f}")
    #
    # print("\nFinal Results (10 runs) for 5-shot:")
    # print(f"Macro-F1: Mean = {f1_macro_5_mean:.4f}, Std = {f1_macro_5_std:.4f}")

# t=0.1, {dataset}_3.pt
# TwitterCOVID19低
# Macro-F1_mean: 0.6189 var: 0.0261  Micro-F1_mean: 0.6691 var: 0.0183 auc 0.6173 var: 0.0267
# Macro-F1_mean: 0.5908 var: 0.0401  Micro-F1_mean: 0.6400 var: 0.0261 auc 0.5980 var: 0.0271
# Weibo高
# Macro-F1_mean: 0.8045 var: 0.0050  Micro-F1_mean: 0.8045 var: 0.0046 auc 0.8358 var: 0.0045
# Macro-F1_mean: 0.7903 var: 0.0080  Micro-F1_mean: 0.7916 var: 0.0072 auc 0.7995 var: 0.0105
# WeiboCOVID19高
# Macro-F1_mean: 0.5313 var: 0.0947  Micro-F1_mean: 0.5512 var: 0.0838 auc 0.5450 var: 0.1001
# Macro-F1_mean: 0.6738 var: 0.0169  Micro-F1_mean: 0.6682 var: 0.0246 auc 0.7095 var: 0.0294
# PHEME低一点但是SOTA
# Macro-F1_mean: 0.5699 var: 0.0115  Micro-F1_mean: 0.6368 var: 0.0061 auc 0.5632 var: 0.0136
# Macro-F1_mean: 0.6072 var: 0.0197  Micro-F1_mean: 0.6704 var: 0.0055 auc 0.6072 var: 0.0045
# DRWeiboV3低和RAGCL持平
# Macro-F1_mean: 0.5169 var: 0.0207  Micro-F1_mean: 0.5308 var: 0.0149 auc 0.5334 var: 0.0290
# Macro-F1_mean: 0.5364 var: 0.0072  Micro-F1_mean: 0.5302 var: 0.0074 auc 0.5307 var: 0.0186
# Politifact低
# Macro-F1_mean: 0.4909 var: 0.0526  Micro-F1_mean: 0.5047 var: 0.0480 auc 0.5180 var: 0.0484
# Macro-F1_mean: 0.5931 var: 0.0638  Micro-F1_mean: 0.5971 var: 0.0563 auc 0.6066 var: 0.0397
# Gossipcop差不多
# Macro-F1_mean: 0.7640 var: 0.0172  Micro-F1_mean: 0.7643 var: 0.0155 auc 0.7881 var: 0.0147
# Macro-F1_mean: 0.7582 var: 0.0072  Micro-F1_mean: 0.7610 var: 0.0065 auc 0.7824 var: 0.0082


# t=0.15 {dataset}_4.pt
# TwitterCOVID19
# Macro-F1_mean: 0.6500 var: 0.0230  Micro-F1_mean: 0.6889 var: 0.0153 auc 0.6396 var: 0.0286
# Macro-F1_mean: 0.6057 var: 0.0309  Micro-F1_mean: 0.6410 var: 0.0343 auc 0.6026 var: 0.0443
# Weibo
# Macro-F1_mean: 0.7802 var: 0.0126  Micro-F1_mean: 0.7813 var: 0.0114 auc 0.7791 var: 0.0184
# Macro-F1_mean: 0.7770 var: 0.0089  Micro-F1_mean: 0.7786 var: 0.0084 auc 0.7738 var: 0.0160
# WeiboCOVID19
# Macro-F1_mean: 0.4932 var: 0.0698  Micro-F1_mean: 0.5306 var: 0.0742 auc 0.5249 var: 0.1048
# Macro-F1_mean: 0.6318 var: 0.0315  Micro-F1_mean: 0.6429 var: 0.0363 auc 0.6609 var: 0.0894
# PHEME
# Macro-F1_mean: 0.6038 var: 0.0119  Micro-F1_mean: 0.6604 var: 0.0056 auc 0.6108 var: 0.0109
# Macro-F1_mean: 0.5954 var: 0.0146  Micro-F1_mean: 0.6530 var: 0.0201 auc 0.5923 var: 0.0392
# DRWeiboV3
# Macro-F1_mean: 0.5383 var: 0.0238  Micro-F1_mean: 0.5476 var: 0.0093 auc 0.5611 var: 0.0111
# Macro-F1_mean: 0.5306 var: 0.0246  Micro-F1_mean: 0.5329 var: 0.0182 auc 0.5456 var: 0.0347
# 256dim
# DRWeiboV3
# Macro-F1_mean: 0.5468 var: 0.0200  Micro-F1_mean: 0.5535 var: 0.0131 auc 0.5632 var: 0.0178
# Macro-F1_mean: 0.5317 var: 0.0135  Micro-F1_mean: 0.5339 var: 0.0110 auc 0.5291 var: 0.0179
# Politifact
# Macro-F1_mean: 0.4920 var: 0.0732  Micro-F1_mean: 0.5241 var: 0.0505 auc 0.5419 var: 0.0495
# Macro-F1_mean: 0.6185 var: 0.0200  Micro-F1_mean: 0.6206 var: 0.0219 auc 0.6263 var: 0.0115
# Max
# Politifact
# Macro-F1_mean: 0.5006 var: 0.0522  Micro-F1_mean: 0.5212 var: 0.0384 auc 0.5438 var: 0.0530
# Macro-F1_mean: 0.5781 var: 0.0518  Micro-F1_mean: 0.5657 var: 0.0416 auc 0.5941 var: 0.0278
# Gossipcop
# Macro-F1_mean: 0.6868 var: 0.0084  Micro-F1_mean: 0.6892 var: 0.0033 auc 0.6962 var: 0.0021
# Macro-F1_mean: 0.6960 var: 0.0145  Micro-F1_mean: 0.6876 var: 0.0234 auc 0.6995 var: 0.0216
# Max
# Gossipcop
# Macro-F1_mean: 0.7668 var: 0.0052  Micro-F1_mean: 0.7674 var: 0.0046 auc 0.7882 var: 0.0060
# Macro-F1_mean: 0.7641 var: 0.0041  Micro-F1_mean: 0.7655 var: 0.0028 auc 0.7848 var: 0.0034


# t=0.13 {dataset}_6.pt
# TwitterCOVID19
# Macro-F1_mean: 0.6152 var: 0.0479  Micro-F1_mean: 0.6789 var: 0.0237 auc 0.6252 var: 0.0342
# Macro-F1_mean: 0.6133 var: 0.0329  Micro-F1_mean: 0.6324 var: 0.0287 auc 0.6274 var: 0.0427
# Weibo
# Macro-F1_mean: 0.7818 var: 0.0103  Micro-F1_mean: 0.7831 var: 0.0099 auc 0.7813 var: 0.0180
# Macro-F1_mean: 0.7820 var: 0.0103  Micro-F1_mean: 0.7836 var: 0.0094 auc 0.7820 var: 0.0108
# WeiboCOVID19
# Macro-F1_mean: 0.5678 var: 0.0385  Micro-F1_mean: 0.5552 var: 0.0540 auc 0.5708 var: 0.0835
# Macro-F1_mean: 0.6224 var: 0.0128  Micro-F1_mean: 0.6246 var: 0.0290 auc 0.6939 var: 0.0175
# PHEME
# Macro-F1_mean: 0.4893 var: 0.0122  Micro-F1_mean: 0.6303 var: 0.0079 auc 0.5133 var: 0.0230
# Macro-F1_mean: 0.6390 var: 0.0076  Micro-F1_mean: 0.6790 var: 0.0052 auc 0.6363 var: 0.0114

# t=0.15, u=0.4, {dataset}_7.pt
# TwitterCOVID19
# Macro-F1_mean: 0.6187 var: 0.0551  Micro-F1_mean: 0.6587 var: 0.0379 auc 0.6002 var: 0.0438
# Macro-F1_mean: 0.5864 var: 0.0387  Micro-F1_mean: 0.6134 var: 0.0241 auc 0.5865 var: 0.0701
# Weibo
# Macro-F1_mean: 0.5736 var: 0.0506  Micro-F1_mean: 0.5880 var: 0.0354 auc 0.5845 var: 0.0392
# Macro-F1_mean: 0.6918 var: 0.0083  Micro-F1_mean: 0.6945 var: 0.0084 auc 0.7091 var: 0.0097
# WeiboCOVID19
# Macro-F1_mean: 0.4944 var: 0.0994  Micro-F1_mean: 0.5172 var: 0.0896 auc 0.5100 var: 0.1162
# Macro-F1_mean: 0.6422 var: 0.0314  Micro-F1_mean: 0.6498 var: 0.0406 auc 0.6988 var: 0.0444
# PHEME
# Macro-F1_mean: 0.5605 var: 0.0203  Micro-F1_mean: 0.6166 var: 0.0359 auc 0.5690 var: 0.0210
# Macro-F1_mean: 0.5985 var: 0.0139  Micro-F1_mean: 0.6587 var: 0.0080 auc 0.5935 var: 0.0204


# t=0.09, u=0.6, {dataset}_5.pt
# TwitterCOVID19
# Macro-F1_mean: 0.4178 var: 0.0527  Micro-F1_mean: 0.5342 var: 0.0766 auc 0.4648 var: 0.0556
# Macro-F1_mean: 0.5932 var: 0.0280  Micro-F1_mean: 0.6434 var: 0.0213 auc 0.6411 var: 0.0462
# Weibo
# Macro-F1_mean: 0.7540 var: 0.0256  Micro-F1_mean: 0.7542 var: 0.0258 auc 0.7727 var: 0.0191
# Macro-F1_mean: 0.7769 var: 0.0175  Micro-F1_mean: 0.7787 var: 0.0158 auc 0.7717 var: 0.0185
# WeiboCOVID19
# Macro-F1_mean: 0.4541 var: 0.0614  Micro-F1_mean: 0.4687 var: 0.0578 auc 0.5159 var: 0.1352
# Macro-F1_mean: 0.6141 var: 0.0227  Micro-F1_mean: 0.6183 var: 0.0237 auc 0.6708 var: 0.0346
# PHEME
# Macro-F1_mean: 0.5877 var: 0.0277  Micro-F1_mean: 0.6623 var: 0.0076 auc 0.5969 var: 0.0188
# Macro-F1_mean: 0.6343 var: 0.0148  Micro-F1_mean: 0.6737 var: 0.0151 auc 0.6358 var: 0.0175


# _8.pt, 正确的 0.1
# TwitterCOVID19
# Macro-F1_mean: 0.5954 var: 0.0413  Micro-F1_mean: 0.6836 var: 0.0189 auc 0.6188 var: 0.0153
# Macro-F1_mean: 0.6327 var: 0.0341  Micro-F1_mean: 0.6793 var: 0.0309 auc 0.6446 var: 0.0268
# WeiboCOVID19降
# Macro-F1_mean: 0.5336 var: 0.0409  Micro-F1_mean: 0.5236 var: 0.0466 auc 0.5271 var: 0.1033
# Macro-F1_mean: 0.6398 var: 0.0248  Micro-F1_mean: 0.6522 var: 0.0258 auc 0.6728 var: 0.0621
# PHEME
# Macro-F1_mean: 0.6283 var: 0.0126  Micro-F1_mean: 0.6773 var: 0.0234 auc 0.6484 var: 0.0102
# Macro-F1_mean: 0.6536 var: 0.0106  Micro-F1_mean: 0.6960 var: 0.0121 auc 0.6713 var: 0.0163
# DRWeiboV3
# Macro-F1_mean: 0.5571 var: 0.0329  Micro-F1_mean: 0.5704 var: 0.0260 auc 0.5939 var: 0.0190
# Macro-F1_mean: 0.5743 var: 0.0143  Micro-F1_mean: 0.5630 var: 0.0233 auc 0.5906 var: 0.0284
# Weibo降
# Macro-F1_mean: 0.5867 var: 0.0865  Micro-F1_mean: 0.6147 var: 0.0526 auc 0.6345 var: 0.0616
# Macro-F1_mean: 0.6798 var: 0.0094  Micro-F1_mean: 0.6806 var: 0.0098 auc 0.6890 var: 0.0137
# Gossipcop
# Macro-F1_mean: 0.7694 var: 0.0075  Micro-F1_mean: 0.7696 var: 0.0070 auc 0.7968 var: 0.0064
# Macro-F1_mean: 0.7672 var: 0.0055  Micro-F1_mean: 0.7679 var: 0.0050 auc 0.8000 var: 0.0068
# Politifact
# Macro-F1_mean: 0.5073 var: 0.0409  Micro-F1_mean: 0.5104 var: 0.0336 auc 0.5471 var: 0.0291
# Macro-F1_mean: 0.6351 var: 0.0262  Micro-F1_mean: 0.6172 var: 0.0569 auc 0.6427 var: 0.0317


# t=0.1, {dataset}_3.pt
# TwitterCOVID19
# Macro-F1_mean: 0.6189 var: 0.0261  Micro-F1_mean: 0.6691 var: 0.0183 auc 0.6173 var: 0.0267
# Macro-F1_mean: 0.5908 var: 0.0401  Micro-F1_mean: 0.6400 var: 0.0261 auc 0.5980 var: 0.0271
# Weibo
# Macro-F1_mean: 0.8045 var: 0.0050  Micro-F1_mean: 0.8045 var: 0.0046 auc 0.8358 var: 0.0045
# Macro-F1_mean: 0.7903 var: 0.0080  Micro-F1_mean: 0.7916 var: 0.0072 auc 0.7995 var: 0.0105
# WeiboCOVID19
# Macro-F1_mean: 0.5313 var: 0.0947  Micro-F1_mean: 0.5512 var: 0.0838 auc 0.5450 var: 0.1001
# Macro-F1_mean: 0.6738 var: 0.0169  Micro-F1_mean: 0.6682 var: 0.0246 auc 0.7095 var: 0.0294
# PHEME
# Macro-F1_mean: 0.5699 var: 0.0115  Micro-F1_mean: 0.6368 var: 0.0061 auc 0.5632 var: 0.0136
# Macro-F1_mean: 0.6072 var: 0.0197  Micro-F1_mean: 0.6704 var: 0.0055 auc 0.6072 var: 0.0045
# DRWeiboV3
# Macro-F1_mean: 0.5169 var: 0.0207  Micro-F1_mean: 0.5308 var: 0.0149 auc 0.5334 var: 0.0290
# Macro-F1_mean: 0.5364 var: 0.0072  Micro-F1_mean: 0.5302 var: 0.0074 auc 0.5307 var: 0.0186
# Politifact
# Macro-F1_mean: 0.4909 var: 0.0526  Micro-F1_mean: 0.5047 var: 0.0480 auc 0.5180 var: 0.0484
# Macro-F1_mean: 0.5931 var: 0.0638  Micro-F1_mean: 0.5971 var: 0.0563 auc 0.6066 var: 0.0397
# Gossipcop
# Macro-F1_mean: 0.7640 var: 0.0172  Micro-F1_mean: 0.7643 var: 0.0155 auc 0.7881 var: 0.0147
# Macro-F1_mean: 0.7582 var: 0.0072  Micro-F1_mean: 0.7610 var: 0.0065 auc 0.7824 var: 0.0082


# 0.15, _9.pt
# Weibo
# Macro-F1_mean: 0.6090 var: 0.0721  Micro-F1_mean: 0.6261 var: 0.0516 auc 0.6330 var: 0.0557
# Macro-F1_mean: 0.7146 var: 0.0246  Micro-F1_mean: 0.7152 var: 0.0249 auc 0.7256 var: 0.0289
# DRWeiboV3
# Macro-F1_mean: 0.5765 var: 0.0162  Micro-F1_mean: 0.5835 var: 0.0135 auc 0.6018 var: 0.0063
# Macro-F1_mean: 0.5829 var: 0.0097  Micro-F1_mean: 0.5751 var: 0.0218 auc 0.5977 var: 0.0295
# WeiboCOVID19
# Macro-F1_mean: 0.5036 var: 0.0515  Micro-F1_mean: 0.5455 var: 0.0503 auc 0.4860 var: 0.1133
# Macro-F1_mean: 0.5791 var: 0.0216  Micro-F1_mean: 0.5837 var: 0.0271 auc 0.6326 var: 0.0404
# PHEME
# Macro-F1_mean: 0.6237 var: 0.0128  Micro-F1_mean: 0.6735 var: 0.0094 auc 0.6249 var: 0.0155
# Macro-F1_mean: 0.6341 var: 0.0171  Micro-F1_mean: 0.6734 var: 0.0137 auc 0.6269 var: 0.0101
# TwitterCOVID19
# Macro-F1_mean: 0.6294 var: 0.0153  Micro-F1_mean: 0.6815 var: 0.0154 auc 0.6313 var: 0.0126
# Macro-F1_mean: 0.5876 var: 0.0383  Micro-F1_mean: 0.6279 var: 0.0335 auc 0.5826 var: 0.0599


# 10.pt  p(x)*x
# DRWeiboV3
# Macro-F1_mean: 0.5335 var: 0.0171  Micro-F1_mean: 0.5340 var: 0.0151 auc 0.5298 var: 0.0151
# Macro-F1_mean: 0.6280 var: 0.0129  Micro-F1_mean: 0.6325 var: 0.0115 auc 0.6624 var: 0.0134
# PHEME
# Macro-F1_mean: 0.5266 var: 0.0272  Micro-F1_mean: 0.5853 var: 0.0444 auc 0.5307 var: 0.0531
# Macro-F1_mean: 0.5543 var: 0.0232  Micro-F1_mean: 0.6108 var: 0.0155 auc 0.5546 var: 0.0359
# WeiboCOVID19
# Macro-F1_mean: 0.5509 var: 0.0566  Micro-F1_mean: 0.5731 var: 0.0479 auc 0.5217 var: 0.1020
# Macro-F1_mean: 0.6231 var: 0.0272  Micro-F1_mean: 0.6353 var: 0.0377 auc 0.6798 var: 0.0363
# TwitterCOVID19
# Macro-F1_mean: 0.5911 var: 0.0672  Micro-F1_mean: 0.6238 var: 0.0751 auc 0.6112 var: 0.0351
# Macro-F1_mean: 0.5931 var: 0.0479  Micro-F1_mean: 0.6197 var: 0.0561 auc 0.6111 var: 0.0543

# 11.pt  p(root)*x
# TwitterCOVID19
# Macro-F1_mean: 0.6071 var: 0.0246  Micro-F1_mean: 0.6456 var: 0.0426 auc 0.6163 var: 0.0291
# Macro-F1_mean: 0.6486 var: 0.0269  Micro-F1_mean: 0.6738 var: 0.0455 auc 0.6398 var: 0.0400
# PHEME
# Macro-F1_mean: 0.5128 var: 0.0455  Micro-F1_mean: 0.5587 var: 0.0580 auc 0.5200 var: 0.0440
# Macro-F1_mean: 0.6110 var: 0.0062  Micro-F1_mean: 0.6443 var: 0.0202 auc 0.6239 var: 0.0539
# WeiboCOVID19
# Macro-F1_mean: 0.4713 var: 0.1095  Micro-F1_mean: 0.5300 var: 0.1074 auc 0.4543 var: 0.1400
# Macro-F1_mean: 0.6252 var: 0.0513  Micro-F1_mean: 0.6370 var: 0.0619 auc 0.6711 var: 0.0406
# DRWeiboV3
# Macro-F1_mean: 0.5840 var: 0.0177  Micro-F1_mean: 0.5889 var: 0.0139 auc 0.6111 var: 0.0192
# Macro-F1_mean: 0.5357 var: 0.0088  Micro-F1_mean: 0.5415 var: 0.0120 auc 0.5525 var: 0.0149


# 13.pt p(root)+x
# TwitterCOVID19
# Macro-F1_mean: 0.6319 var: 0.0271  Micro-F1_mean: 0.6849 var: 0.0129 auc 0.6186 var: 0.0147
# Macro-F1_mean: 0.6223 var: 0.0254  Micro-F1_mean: 0.6693 var: 0.0119 auc 0.6169 var: 0.0185
# PHEME
# Macro-F1_mean: 0.4472 var: 0.0597  Micro-F1_mean: 0.5918 var: 0.0885 auc 0.5024 var: 0.0112
# Macro-F1_mean: 0.4888 var: 0.0143  Micro-F1_mean: 0.6313 var: 0.0001 auc 0.5035 var: 0.0069
# WeiboCOVID19
# Macro-F1_mean: 0.4922 var: 0.0553  Micro-F1_mean: 0.5013 var: 0.0621 auc 0.5264 var: 0.0766
# Macro-F1_mean: 0.6148 var: 0.0614  Micro-F1_mean: 0.6311 var: 0.0369 auc 0.7162 var: 0.0657
# DRWeiboV3
# Macro-F1_mean: 0.5032 var: 0.0214  Micro-F1_mean: 0.5105 var: 0.0112 auc 0.5107 var: 0.0198
# Macro-F1_mean: 0.5273 var: 0.0167  Micro-F1_mean: 0.5222 var: 0.0198 auc 0.5307 var: 0.0257


# 12.pt w/o alpha
# PHEME
# Macro-F1_mean: 0.4209 var: 0.0369  Micro-F1_mean: 0.5151 var: 0.1013 auc 0.4578 var: 0.0658
# Macro-F1_mean: 0.6365 var: 0.0140  Micro-F1_mean: 0.6703 var: 0.0133 auc 0.6437 var: 0.0142
# DRWeiboV3
# Macro-F1_mean: 0.5812 var: 0.0322  Micro-F1_mean: 0.5923 var: 0.0157 auc 0.5913 var: 0.0212
# Macro-F1_mean: 0.5657 var: 0.0078  Micro-F1_mean: 0.5534 var: 0.0170 auc 0.5516 var: 0.0232
# TwitterCOVID19
# Macro-F1_mean: 0.6364 var: 0.0451  Micro-F1_mean: 0.6674 var: 0.0530 auc 0.6352 var: 0.0284
# Macro-F1_mean: 0.6217 var: 0.0344  Micro-F1_mean: 0.6400 var: 0.0356 auc 0.6365 var: 0.0283
# WeiboCOVID19
# Macro-F1_mean: 0.3725 var: 0.0555  Micro-F1_mean: 0.4303 var: 0.0414 auc 0.4763 var: 0.0592
# Macro-F1_mean: 0.5364 var: 0.0302  Micro-F1_mean: 0.5509 var: 0.0298 auc 0.5486 var: 0.0562
# Poli
# Macro-F1_mean: 0.4698 var: 0.0290  Micro-F1_mean: 0.4986 var: 0.0175 auc 0.5175 var: 0.0142
# Macro-F1_mean: 0.6422 var: 0.0259  Micro-F1_mean: 0.6319 var: 0.0281 auc 0.6692 var: 0.0246

# 14.pt x
# TwitterCOVID19
# Macro-F1_mean: 0.5399 var: 0.0442  Micro-F1_mean: 0.6003 var: 0.0246 auc 0.5399 var: 0.0427
# Macro-F1_mean: 0.6309 var: 0.0319  Micro-F1_mean: 0.6459 var: 0.0403 auc 0.6268 var: 0.0520
# PHEME
# Macro-F1_mean: 0.5549 var: 0.0545  Micro-F1_mean: 0.6328 var: 0.0581 auc 0.5828 var: 0.0237
# Macro-F1_mean: 0.6495 var: 0.0152  Micro-F1_mean: 0.6769 var: 0.0113 auc 0.6672 var: 0.0319
# DRWeiboV3
# Macro-F1_mean: 0.5415 var: 0.0269  Micro-F1_mean: 0.5537 var: 0.0181 auc 0.5720 var: 0.0241
# Macro-F1_mean: 0.6165 var: 0.0179  Micro-F1_mean: 0.6216 var: 0.0182 auc 0.6542 var: 0.0222
# WeiboCOVID19
# Macro-F1_mean: 0.4046 var: 0.0761  Micro-F1_mean: 0.4939 var: 0.1028 auc 0.4713 var: 0.0927
# Macro-F1_mean: 0.4872 var: 0.0331  Micro-F1_mean: 0.4855 var: 0.0397 auc 0.5406 var: 0.0362


# 15.pt, 0.05
# TwitterCOVID19
# Macro-F1_mean: 0.6437 var: 0.0511  Micro-F1_mean: 0.7007 var: 0.0317 auc 0.6468 var: 0.0418
# Macro-F1_mean: 0.6261 var: 0.0439  Micro-F1_mean: 0.6776 var: 0.0283 auc 0.6232 var: 0.0310
# PHEME
# Macro-F1_mean: 0.5439 var: 0.0399  Micro-F1_mean: 0.5891 var: 0.0460 auc 0.5312 var: 0.0672
# Macro-F1_mean: 0.5666 var: 0.0153  Micro-F1_mean: 0.5793 var: 0.0276 auc 0.5637 var: 0.0383
# WeiboCOVID19
# Macro-F1_mean: 0.4212 var: 0.0637  Micro-F1_mean: 0.4391 var: 0.0844 auc 0.4393 var: 0.1206
# Macro-F1_mean: 0.5361 var: 0.0499  Micro-F1_mean: 0.5391 var: 0.0495 auc 0.5934 var: 0.0301
# DRWeiboV3
# Macro-F1_mean: 0.5451 var: 0.0281  Micro-F1_mean: 0.5555 var: 0.0215 auc 0.5730 var: 0.0161
# Macro-F1_mean: 0.5625 var: 0.0124  Micro-F1_mean: 0.5541 var: 0.0228 auc 0.5662 var: 0.0346
# Weibo
# Macro-F1_mean: 0.6551 var: 0.0422  Micro-F1_mean: 0.6562 var: 0.0407 auc 0.6820 var: 0.0334
# Macro-F1_mean: 0.6996 var: 0.0123  Micro-F1_mean: 0.6989 var: 0.0120 auc 0.7149 var: 0.0182


# 16.pt  0.2
# DRWeiboV3
# Macro-F1_mean: 0.5829 var: 0.0219  Micro-F1_mean: 0.5916 var: 0.0157 auc 0.6126 var: 0.0167
# Macro-F1_mean: 0.5465 var: 0.0114  Micro-F1_mean: 0.5438 var: 0.0185 auc 0.5532 var: 0.0184
# WeiboCOVID19
# Macro-F1_mean: 0.3997 var: 0.1269  Micro-F1_mean: 0.4529 var: 0.1167 auc 0.4246 var: 0.1563
# Macro-F1_mean: 0.5863 var: 0.0527  Micro-F1_mean: 0.5910 var: 0.0538 auc 0.6406 var: 0.0639
# PHEME
# Macro-F1_mean: 0.5500 var: 0.0339  Micro-F1_mean: 0.6111 var: 0.0464 auc 0.5581 var: 0.0235
# Macro-F1_mean: 0.6023 var: 0.0186  Micro-F1_mean: 0.6503 var: 0.0155 auc 0.5893 var: 0.0406

# 17.pt
# TwitterCOVID19
# Macro-F1_mean: 0.5639 var: 0.0103  Micro-F1_mean: 0.5970 var: 0.0134 auc 0.5706 var: 0.0102
# Macro-F1_mean: 0.5584 var: 0.0360  Micro-F1_mean: 0.6190 var: 0.0099 auc 0.5236 var: 0.0147
# PHEME
# Macro-F1_mean: 0.5316 var: 0.0683  Micro-F1_mean: 0.5438 var: 0.0824 auc 0.5622 var: 0.0714
# Macro-F1_mean: 0.6331 var: 0.0077  Micro-F1_mean: 0.6784 var: 0.0061 auc 0.6563 var: 0.0101
# WeiboCOVID19
# Macro-F1_mean: 0.4462 var: 0.0521  Micro-F1_mean: 0.5461 var: 0.0688 auc 0.4904 var: 0.0667
# Macro-F1_mean: 0.4068 var: 0.0717  Micro-F1_mean: 0.4526 var: 0.1000 auc 0.5049 var: 0.0288
# DRWeiboV3
# Macro-F1_mean: 0.4682 var: 0.0116  Micro-F1_mean: 0.5207 var: 0.0046 auc 0.4967 var: 0.0031
# Macro-F1_mean: 0.5164 var: 0.0038  Micro-F1_mean: 0.5290 var: 0.0064 auc 0.5190 var: 0.0063

# 18.pt
# TwitterCOVID19
# Macro-F1_mean: 0.6223 var: 0.0156  Micro-F1_mean: 0.6711 var: 0.0104 auc 0.6128 var: 0.0215
# Macro-F1_mean: 0.5873 var: 0.0245  Micro-F1_mean: 0.6555 var: 0.0106 auc 0.5923 var: 0.0211
# PHEME
# Macro-F1_mean: 0.4807 var: 0.0189  Micro-F1_mean: 0.6036 var: 0.0591 auc 0.5400 var: 0.0394
# Macro-F1_mean: 0.5006 var: 0.0089  Micro-F1_mean: 0.6302 var: 0.0039 auc 0.5667 var: 0.0091
# WeiboCOVID19
# Macro-F1_mean: 0.4881 var: 0.0543  Micro-F1_mean: 0.5603 var: 0.0526 auc 0.5462 var: 0.0765
# Macro-F1_mean: 0.6290 var: 0.0234  Micro-F1_mean: 0.6592 var: 0.0253 auc 0.6305 var: 0.0774
# DRWeiboV3
# Macro-F1_mean: 0.4586 var: 0.0102  Micro-F1_mean: 0.5204 var: 0.0110 auc 0.4893 var: 0.0148
# Macro-F1_mean: 0.5542 var: 0.0121  Micro-F1_mean: 0.5657 var: 0.0075 auc 0.5735 var: 0.0109

# SAGE
# TwitterCOVID19
# Macro-F1_mean: 0.5425 var: 0.0296  Micro-F1_mean: 0.6339 var: 0.0198 auc 0.5333 var: 0.0379
# Macro-F1_mean: 0.5973 var: 0.0252  Micro-F1_mean: 0.6497 var: 0.0293 auc 0.6022 var: 0.0361
# PHEME
# Macro-F1_mean: 0.5080 var: 0.0441  Micro-F1_mean: 0.6362 var: 0.0042 auc 0.5377 var: 0.0220
# Macro-F1_mean: 0.6153 var: 0.0148  Micro-F1_mean: 0.6406 var: 0.0145 auc 0.5503 var: 0.0261
# DRWeiboV3
# Macro-F1_mean: 0.6428 var: 0.0298  Micro-F1_mean: 0.6488 var: 0.0277 auc 0.6791 var: 0.0324
# Macro-F1_mean: 0.6739 var: 0.0113  Micro-F1_mean: 0.6737 var: 0.0116 auc 0.7117 var: 0.0078
# WeiboCOVID19
# Macro-F1_mean: 0.4718 var: 0.0938  Micro-F1_mean: 0.4865 var: 0.0879 auc 0.4954 var: 0.0863
# Macro-F1_mean: 0.5963 var: 0.0240  Micro-F1_mean: 0.6131 var: 0.0178 auc 0.6305 var: 0.0166

# GAT
# DRWeiboV3
# Macro-F1_mean: 0.5423 var: 0.0316  Micro-F1_mean: 0.5547 var: 0.0207 auc 0.5783 var: 0.0282
# Macro-F1_mean: 0.5916 var: 0.0080  Micro-F1_mean: 0.5920 var: 0.0085 auc 0.6178 var: 0.0098
# WeiboCOVID19
# Macro-F1_mean: 0.5209 var: 0.1023  Micro-F1_mean: 0.5387 var: 0.0989 auc 0.5488 var: 0.1516
# Macro-F1_mean: 0.5660 var: 0.0459  Micro-F1_mean: 0.5761 var: 0.0501 auc 0.6386 var: 0.0680
# PHEME
# Macro-F1_mean: 0.4268 var: 0.0569  Micro-F1_mean: 0.4423 var: 0.0415 auc 0.4841 var: 0.0289
# Macro-F1_mean: 0.6260 var: 0.0193  Micro-F1_mean: 0.6537 var: 0.0199 auc 0.6445 var: 0.0373
# TwitterCOVID19
# Macro-F1_mean: 0.6386 var: 0.0198  Micro-F1_mean: 0.6886 var: 0.0149 auc 0.6396 var: 0.0217
# Macro-F1_mean: 0.5904 var: 0.0539  Micro-F1_mean: 0.6366 var: 0.0533 auc 0.5809 var: 0.0600


# 256
# DRWeiboV3
# Macro-F1_mean: 0.5422 var: 0.0179  Micro-F1_mean: 0.5516 var: 0.0117 auc 0.5721 var: 0.0141
# Macro-F1_mean: 0.5687 var: 0.0031  Micro-F1_mean: 0.5593 var: 0.0180 auc 0.5696 var: 0.0334
# WeiboCOVID19
# Macro-F1_mean: 0.5100 var: 0.0351  Micro-F1_mean: 0.4990 var: 0.0546 auc 0.5868 var: 0.0892
# Macro-F1_mean: 0.6601 var: 0.0089  Micro-F1_mean: 0.6713 var: 0.0086 auc 0.7000 var: 0.0146


# WeiboCOVID19
# Macro-F1_mean: 0.4024 var: 0.0200  Micro-F1_mean: 0.5037 var: 0.0895 auc 0.4457 var: 0.0629
# Macro-F1_mean: 0.6906 var: 0.0277  Micro-F1_mean: 0.6900 var: 0.0560 auc 0.7237 var: 0.0430
# PHEME
# Macro-F1_mean: 0.4909 var: 0.0287  Micro-F1_mean: 0.5149 var: 0.0561 auc 0.5156 var: 0.0253
# Macro-F1_mean: 0.6094 var: 0.0331  Micro-F1_mean: 0.6515 var: 0.0198 auc 0.5912 var: 0.0421
# Politifact
# Macro-F1_mean: 0.4384 var: 0.0452  Micro-F1_mean: 0.5005 var: 0.0148 auc 0.5273 var: 0.0107
# Macro-F1_mean: 0.5957 var: 0.0566  Micro-F1_mean: 0.5618 var: 0.0619 auc 0.5887 var: 0.0421
# DRWeiboV3
# Macro-F1_mean: 0.5225 var: 0.0129  Micro-F1_mean: 0.5319 var: 0.0172 auc 0.5346 var: 0.0254
# Macro-F1_mean: 0.4568 var: 0.0232  Micro-F1_mean: 0.5238 var: 0.0068 auc 0.5029 var: 0.0135
# Weibo
# Macro-F1_mean: 0.5548 var: 0.0726  Micro-F1_mean: 0.5821 var: 0.0487 auc 0.5875 var: 0.0755
# Macro-F1_mean: 0.7043 var: 0.0126  Micro-F1_mean: 0.7045 var: 0.0128 auc 0.7171 var: 0.0135
# TwitterCOVID19
# Macro-F1_mean: 0.5886 var: 0.0614  Micro-F1_mean: 0.6604 var: 0.0168 auc 0.5946 var: 0.0218
# Macro-F1_mean: 0.5983 var: 0.0277  Micro-F1_mean: 0.6300 var: 0.0344 auc 0.5634 var: 0.0464


# -------------------------------------------------------------------------------------------
# PHEME
# X+A
# Macro-F1_mean: 0.6028 var: 0.0552  Micro-F1_mean: 0.6683 var: 0.0330 auc 0.6636 var: 0.0851
# Macro-F1_mean: 0.6828 var: 0.0251  Micro-F1_mean: 0.7141 var: 0.0322 auc 0.7642 var: 0.0358
# DRWeiboV3
# Macro-F1_mean: 0.5441 var: 0.0241  Micro-F1_mean: 0.5462 var: 0.0247 auc 0.5556 var: 0.0296
# Macro-F1_mean: 0.5913 var: 0.0188  Micro-F1_mean: 0.5917 var: 0.0187 auc 0.6175 var: 0.0260
# Weibo
# Macro-F1_mean: 0.6300 var: 0.0300  Micro-F1_mean: 0.6349 var: 0.0255 auc 0.6406 var: 0.0306
# Macro-F1_mean: 0.7402 var: 0.0054  Micro-F1_mean: 0.7410 var: 0.0056 auc 0.7479 var: 0.0058
# WeiboCOVID19
# Macro-F1_mean: 0.4613 var: 0.0980  Micro-F1_mean: 0.5276 var: 0.0901 auc 0.5360 var: 0.0871
# Macro-F1_mean: 0.6594 var: 0.0270  Micro-F1_mean: 0.6630 var: 0.0278 auc 0.6873 var: 0.0252
# TwitterCOVID19
# Macro-F1_mean: 0.5641 var: 0.0649  Micro-F1_mean: 0.6144 var: 0.0655 auc 0.5599 var: 0.0522
# Macro-F1_mean: 0.5960 var: 0.0400  Micro-F1_mean: 0.6338 var: 0.0237 auc 0.5663 var: 0.0414


# 最后一层
# DRWeiboV3
# Macro-F1_mean: 0.5546 var: 0.0119  Micro-F1_mean: 0.5479 var: 0.0194 auc 0.5589 var: 0.0115
# Macro-F1_mean: 0.5789 var: 0.0163  Micro-F1_mean: 0.5768 var: 0.0208 auc 0.6055 var: 0.0272
# WeiboCOVID19
# Macro-F1_mean: 0.4710 var: 0.0590  Micro-F1_mean: 0.5118 var: 0.0601 auc 0.4812 var: 0.0746
# Macro-F1_mean: 0.5993 var: 0.0852  Micro-F1_mean: 0.6035 var: 0.0962 auc 0.6452 var: 0.0670
# PHEME
# Macro-F1_mean: 0.5177 var: 0.0261  Micro-F1_mean: 0.6152 var: 0.0411 auc 0.5324 var: 0.0200
# Macro-F1_mean: 0.5959 var: 0.0205  Micro-F1_mean: 0.6527 var: 0.0134 auc 0.5898 var: 0.0354
# Weibo
# Macro-F1_mean: 0.6344 var: 0.0692  Micro-F1_mean: 0.6363 var: 0.0591 auc 0.6404 var: 0.0711
# Macro-F1_mean: 0.7109 var: 0.0251  Micro-F1_mean: 0.7121 var: 0.0251 auc 0.7197 var: 0.0247
# TwitterCOVID19
# Macro-F1_mean: 0.6201 var: 0.0588  Micro-F1_mean: 0.6550 var: 0.0645 auc 0.6207 var: 0.0497
# Macro-F1_mean: 0.6629 var: 0.0304  Micro-F1_mean: 0.6828 var: 0.0265 auc 0.6564 var: 0.0366


# 1 +
# WeiboCOVID19
# Macro-F1_mean: 0.4098 var: 0.1129  Micro-F1_mean: 0.4845 var: 0.0968 auc 0.5675 var: 0.0571
# Macro-F1_mean: 0.6415 var: 0.0219  Micro-F1_mean: 0.6519 var: 0.0248 auc 0.6924 var: 0.0238
# WeiboCOVID19
# Macro-F1_mean: 0.4703 var: 0.1304  Micro-F1_mean: 0.5418 var: 0.1015 auc 0.5437 var: 0.0782
# Macro-F1_mean: 0.6988 var: 0.0393  Micro-F1_mean: 0.7066 var: 0.0481 auc 0.7429 var: 0.0185
# PHEME
# Macro-F1_mean: 0.5047 var: 0.0274  Micro-F1_mean: 0.6132 var: 0.0456 auc 0.5841 var: 0.0518
# Macro-F1_mean: 0.6414 var: 0.0241  Micro-F1_mean: 0.6726 var: 0.0240 auc 0.7147 var: 0.0452
# DRWeiboV3
# Macro-F1_mean: 0.5866 var: 0.0236  Micro-F1_mean: 0.5868 var: 0.0237 auc 0.6094 var: 0.0254
# Macro-F1_mean: 0.6184 var: 0.0085  Micro-F1_mean: 0.6193 var: 0.0083 auc 0.6582 var: 0.0114
# TwitterCOVID19
# Macro-F1_mean: 0.6333 var: 0.0214  Micro-F1_mean: 0.6664 var: 0.0206 auc 0.6133 var: 0.0372
# Macro-F1_mean: 0.6564 var: 0.0356  Micro-F1_mean: 0.6824 var: 0.0251 auc 0.6411 var: 0.0543
# Politifact
# Macro-F1_mean: 0.4804 var: 0.0604  Micro-F1_mean: 0.4783 var: 0.0345 auc 0.5198 var: 0.0433
# Macro-F1_mean: 0.5667 var: 0.0407  Micro-F1_mean: 0.5108 var: 0.0571 auc 0.5547 var: 0.0527
# Gossipcop
# Macro-F1_mean: 0.6979 var: 0.0242  Micro-F1_mean: 0.6995 var: 0.0241 auc 0.7134 var: 0.0255
# Macro-F1_mean: 0.7093 var: 0.0306  Micro-F1_mean: 0.7122 var: 0.0277 auc 0.7314 var: 0.0324
# Weibo
# Macro-F1_mean: 0.5404 var: 0.0313  Micro-F1_mean: 0.5554 var: 0.0181 auc 0.5505 var: 0.0222
# Macro-F1_mean: 0.7016 var: 0.0230  Micro-F1_mean: 0.7020 var: 0.0233 auc 0.7068 var: 0.0242
# Politifact
# Macro-F1_mean: 0.5265 var: 0.0075  Micro-F1_mean: 0.5282 var: 0.0078 auc 0.5329 var: 0.0109
# Macro-F1_mean: 0.6129 var: 0.0116  Micro-F1_mean: 0.6153 var: 0.0108 auc 0.6398 var: 0.0097



# 评论
# DRWeiboV3
# Macro-F1_mean: 0.5629 var: 0.0150  Micro-F1_mean: 0.5685 var: 0.0150 auc 0.5657 var: 0.0177
# Macro-F1_mean: 0.5860 var: 0.0277  Micro-F1_mean: 0.5799 var: 0.0338 auc 0.5744 var: 0.0437
# WeiboCOVID19
# Macro-F1_mean: 0.4501 var: 0.0516  Micro-F1_mean: 0.5498 var: 0.0647 auc 0.5291 var: 0.0563
# Macro-F1_mean: 0.5676 var: 0.0264  Micro-F1_mean: 0.5761 var: 0.0305 auc 0.5961 var: 0.0179
# PHEME
# Macro-F1_mean: 0.5453 var: 0.0167  Micro-F1_mean: 0.5920 var: 0.0351 auc 0.5607 var: 0.0561
# Macro-F1_mean: 0.6100 var: 0.0175  Micro-F1_mean: 0.6436 var: 0.0264 auc 0.6539 var: 0.0432


# DRWeiboV3
# no Weibo
# Macro-F1_mean: 0.5631 var: 0.0190  Micro-F1_mean: 0.5618 var: 0.0193 auc 0.5718 var: 0.0265
# Macro-F1_mean: 0.5672 var: 0.0081  Micro-F1_mean: 0.5757 var: 0.0058 auc 0.5923 var: 0.0072
# no Weibo WeiboCOVID19
# Macro-F1_mean: 0.5488 var: 0.0389  Micro-F1_mean: 0.5583 var: 0.0234 auc 0.5632 var: 0.0297
# Macro-F1_mean: 0.5711 var: 0.0157  Micro-F1_mean: 0.5684 var: 0.0069 auc 0.5806 var: 0.0064
# no Weibo WeiboCOVID19 PHEME
# Macro-F1_mean: 0.4669 var: 0.0101  Micro-F1_mean: 0.5170 var: 0.0046 auc 0.5044 var: 0.0042
# Macro-F1_mean: 0.5341 var: 0.0088  Micro-F1_mean: 0.5374 var: 0.0096 auc 0.5410 var: 0.0166
# no Weibo WeiboCOVID19 PHEME Politifact
# Macro-F1_mean: 0.4733 var: 0.0109  Micro-F1_mean: 0.5141 var: 0.0039 auc 0.5033 var: 0.0091
# Macro-F1_mean: 0.5487 var: 0.0086  Micro-F1_mean: 0.5487 var: 0.0122 auc 0.5487 var: 0.0251
# no Weibo WeiboCOVID19 PHEME Politifact Gossipcop
# Macro-F1_mean: 0.5167 var: 0.0109  Micro-F1_mean: 0.5315 var: 0.0095 auc 0.5210 var: 0.0104
# Macro-F1_mean: 0.5408 var: 0.0113  Micro-F1_mean: 0.5477 var: 0.0100 auc 0.5432 var: 0.0163

# WeiboCOVID19
# no DRWeiboV3
# Macro-F1_mean: 0.4189 var: 0.1166  Micro-F1_mean: 0.5232 var: 0.1265 auc 0.5536 var: 0.0336
# Macro-F1_mean: 0.5923 var: 0.0563  Micro-F1_mean: 0.6107 var: 0.0649 auc 0.6279 var: 0.0578
# no DRWeiboV3 Weibo
# Macro-F1_mean: 0.4648 var: 0.1055  Micro-F1_mean: 0.5242 var: 0.1040 auc 0.5801 var: 0.0398
# Macro-F1_mean: 0.4717 var: 0.0167  Micro-F1_mean: 0.4993 var: 0.0827 auc 0.4971 var: 0.0306
# no DRWeiboV3 Weibo PHEME
# Macro-F1_mean: 0.5876 var: 0.0096  Micro-F1_mean: 0.6108 var: 0.0098 auc 0.5489 var: 0.0909
# Macro-F1_mean: 0.5887 var: 0.0047  Micro-F1_mean: 0.6035 var: 0.0295 auc 0.6074 var: 0.0749
# no DRWeiboV3 Weibo PHEME Politifact
# Macro-F1_mean: 0.6245 var: 0.0321  Micro-F1_mean: 0.6387 var: 0.0159 auc 0.5418 var: 0.0543
# Macro-F1_mean: 0.6099 var: 0.0484  Micro-F1_mean: 0.6488 var: 0.0151 auc 0.4865 var: 0.0884

# PHEME
# no DRWeiboV3
# Macro-F1_mean: 0.5762 var: 0.0294  Micro-F1_mean: 0.6469 var: 0.0110 auc 0.6435 var: 0.0296
# Macro-F1_mean: 0.5907 var: 0.0080  Micro-F1_mean: 0.6323 var: 0.0176 auc 0.6448 var: 0.0232
# no DRWeiboV3 Weibo
# Macro-F1_mean: 0.4875 var: 0.0177  Micro-F1_mean: 0.6152 var: 0.0583 auc 0.5068 var: 0.0092
# Macro-F1_mean: 0.4903 var: 0.0098  Micro-F1_mean: 0.6313 var: 0.0000 auc 0.5138 var: 0.0057
# no DRWeiboV3 Weibo WeiboCOVID19
# Macro-F1_mean: 0.4878 var: 0.0030  Micro-F1_mean: 0.6346 var: 0.0000 auc 0.5227 var: 0.0097
# Macro-F1_mean: 0.4878 var: 0.0049  Micro-F1_mean: 0.6309 var: 0.0013 auc 0.5246 var: 0.0051
# no DRWeiboV3 Weibo WeiboCOVID19 Politifact

# no DRWeiboV3 Weibo WeiboCOVID19 Politifact Gossipcop
# Macro-F1_mean: 0.5121 var: 0.0091  Micro-F1_mean: 0.6147 var: 0.0276 auc 0.5262 var: 0.0195
# Macro-F1_mean: 0.5002 var: 0.0314  Micro-F1_mean: 0.5815 var: 0.0339 auc 0.5052 var: 0.0203



# no adpater
# DRWeiboV3
# Macro-F1_mean: 0.5025 var: 0.0158  Micro-F1_mean: 0.5189 var: 0.0136 auc 0.5202 var: 0.0158
# Macro-F1_mean: 0.5495 var: 0.0034  Micro-F1_mean: 0.5498 var: 0.0091 auc 0.5459 var: 0.0136
# WeiboCOVID19
# Macro-F1_mean: 0.5000 var: 0.0195  Micro-F1_mean: 0.5310 var: 0.0411 auc 0.5196 var: 0.0206
# Macro-F1_mean: 0.4389 var: 0.0692  Micro-F1_mean: 0.4682 var: 0.0721 auc 0.5544 var: 0.0247
# PHEME
# Macro-F1_mean: 0.4744 var: 0.0369  Micro-F1_mean: 0.5237 var: 0.0733 auc 0.5012 var: 0.0335
# Macro-F1_mean: 0.4670 var: 0.0174  Micro-F1_mean: 0.6289 var: 0.0154 auc 0.5145 var: 0.0167


# 负迁移实验，在DRWeibo上训练
# PHEME
# Macro-F1_mean: 0.5702 var: 0.0414  Micro-F1_mean: 0.6024 var: 0.0547 auc 0.5945 var: 0.0515
# Macro-F1_mean: 0.6118 var: 0.0205  Micro-F1_mean: 0.6519 var: 0.0215 auc 0.6531 var: 0.0217


# 五个数据集RoBERTa 71,1+X,"./{dataset}_1.pt",add
# DRWeiboV3
# Macro-F1_mean: 0.5193 var: 0.0426  Micro-F1_mean: 0.5226 var: 0.0264 auc 0.5245 var: 0.0319
# Macro-F1_mean: 0.6151 var: 0.0113  Micro-F1_mean: 0.6160 var: 0.0114 auc 0.6558 var: 0.0132
# WeiboCOVID19
# Macro-F1_mean: 0.5246 var: 0.0852  Micro-F1_mean: 0.5906 var: 0.0704 auc 0.5612 var: 0.0643
# Macro-F1_mean: 0.6481 var: 0.0261  Micro-F1_mean: 0.6696 var: 0.0260 auc 0.6486 var: 0.0434
# PHEME
# Macro-F1_mean: 0.6063 var: 0.0220  Micro-F1_mean: 0.6611 var: 0.0075 auc 0.6578 var: 0.0117
# Macro-F1_mean: 0.5938 var: 0.0278  Micro-F1_mean: 0.6615 var: 0.0207 auc 0.6694 var: 0.0317
# Weibo
# Macro-F1_mean: 0.5074 var: 0.0634  Micro-F1_mean: 0.5248 var: 0.0591 auc 0.5480 var: 0.0743
# Macro-F1_mean: 0.6502 var: 0.0205  Micro-F1_mean: 0.6513 var: 0.0206 auc 0.6698 var: 0.0231
# TwitterCOVID19
# Macro-F1_mean: 0.5971 var: 0.0388  Micro-F1_mean: 0.6416 var: 0.0502 auc 0.6036 var: 0.0263
# Macro-F1_mean: 0.6246 var: 0.0297  Micro-F1_mean: 0.6638 var: 0.0399 auc 0.6186 var: 0.0517


# MEAN
# DRWeiboV3
# Macro-F1_mean: 0.5322 var: 0.0271  Micro-F1_mean: 0.5398 var: 0.0209 auc 0.5561 var: 0.0289
# Macro-F1_mean: 0.6143 var: 0.0070  Micro-F1_mean: 0.6155 var: 0.0065 auc 0.6701 var: 0.0150
# WeiboCOVID19
# Macro-F1_mean: 0.5594 var: 0.0559  Micro-F1_mean: 0.5956 var: 0.0676 auc 0.6327 var: 0.0701
# Macro-F1_mean: 0.6241 var: 0.0292  Micro-F1_mean: 0.6367 var: 0.0387 auc 0.7157 var: 0.0230
# PHEME
# Macro-F1_mean: 0.5842 var: 0.0241  Micro-F1_mean: 0.6259 var: 0.0347 auc 0.5925 var: 0.0526
# Macro-F1_mean: 0.5757 var: 0.0190  Micro-F1_mean: 0.6508 var: 0.0129 auc 0.6182 var: 0.0449
# Weibo
# Macro-F1_mean: 0.5435 var: 0.1009  Micro-F1_mean: 0.5648 var: 0.0946 auc 0.5719 var: 0.1501
# Macro-F1_mean: 0.6955 var: 0.0153  Micro-F1_mean: 0.6962 var: 0.0160 auc 0.7436 var: 0.0223
# TwitterCOVID19
# Macro-F1_mean: 0.6310 var: 0.0175  Micro-F1_mean: 0.6701 var: 0.0287 auc 0.6842 var: 0.0243
# Macro-F1_mean: 0.5637 var: 0.0488  Micro-F1_mean: 0.6041 var: 0.0560 auc 0.5946 var: 0.0755


# RAGCL
# DRWeiboV3
# Macro-F1_mean: 0.5663 var: 0.0144  Micro-F1_mean: 0.5743 var: 0.0140 auc 0.5892 var: 0.0109
# Macro-F1_mean: 0.5364 var: 0.0169  Micro-F1_mean: 0.5348 var: 0.0188 auc 0.5439 var: 0.0242
# Weibo
# Macro-F1_mean: 0.5912 var: 0.0485  Micro-F1_mean: 0.6014 var: 0.0461 auc 0.6121 var: 0.0586
# Macro-F1_mean: 0.7017 var: 0.0093  Micro-F1_mean: 0.7042 var: 0.0079 auc 0.7298 var: 0.0066
# WeiboCOVID19
# Macro-F1_mean: 0.5224 var: 0.0528  Micro-F1_mean: 0.5586 var: 0.0633 auc 0.5254 var: 0.0398
# Macro-F1_mean: 0.5437 var: 0.0214  Micro-F1_mean: 0.5478 var: 0.0357 auc 0.6102 var: 0.0157
# PHEME
# Macro-F1_mean: 0.5345 var: 0.0403  Micro-F1_mean: 0.5599 var: 0.0562 auc 0.5791 var: 0.0337
# Macro-F1_mean: 0.5855 var: 0.0159  Micro-F1_mean: 0.6159 var: 0.0204 auc 0.6202 var: 0.0251
# TwitterCOVID19
# Macro-F1_mean: 0.6510 var: 0.0272  Micro-F1_mean: 0.6862 var: 0.0309 auc 0.6578 var: 0.0234
# Macro-F1_mean: 0.5784 var: 0.0700  Micro-F1_mean: 0.6214 var: 0.0580 auc 0.6051 var: 0.0677


# GAMC



# 3.pt
# DRWeiboV3
# Macro-F1_mean: 0.5123 var: 0.0249  Micro-F1_mean: 0.5101 var: 0.0152 auc 0.5065 var: 0.0247
# Macro-F1_mean: 0.6111 var: 0.0055  Micro-F1_mean: 0.6107 var: 0.0090 auc 0.6488 var: 0.0149
# WeiboCOVID19
# Macro-F1_mean: 0.5535 var: 0.0739  Micro-F1_mean: 0.6138 var: 0.0597 auc 0.6088 var: 0.0722
# Macro-F1_mean: 0.6888 var: 0.0177  Micro-F1_mean: 0.7024 var: 0.0189 auc 0.7223 var: 0.0414
# PHEME
# Macro-F1_mean: 0.5976 var: 0.0142  Micro-F1_mean: 0.6493 var: 0.0109 auc 0.6460 var: 0.0226
# Macro-F1_mean: 0.6134 var: 0.0171  Micro-F1_mean: 0.6649 var: 0.0161 auc 0.6790 var: 0.0294
# Weibo
# Macro-F1_mean: 0.6058 var: 0.0764  Micro-F1_mean: 0.6187 var: 0.0675 auc 0.6532 var: 0.0661
# Macro-F1_mean: 0.7338 var: 0.0222  Micro-F1_mean: 0.7322 var: 0.0236 auc 0.7625 var: 0.0203
# TwitterCOVID19
# Macro-F1_mean: 0.6004 var: 0.0231  Micro-F1_mean: 0.6416 var: 0.0346 auc 0.6038 var: 0.0168
# Macro-F1_mean: 0.6263 var: 0.0277  Micro-F1_mean: 0.6631 var: 0.0288 auc 0.6044 var: 0.0361


# 只用X的，不用Root
# DRWeiboV3
# Macro-F1_mean: 0.5240 var: 0.0445  Micro-F1_mean: 0.5275 var: 0.0444 auc 0.5365 var: 0.0624
# Macro-F1_mean: 0.6631 var: 0.0091  Micro-F1_mean: 0.6653 var: 0.0094 auc 0.7118 var: 0.0097
# Weibo
# Macro-F1_mean: 0.5118 var: 0.0493  Micro-F1_mean: 0.5442 var: 0.0343 auc 0.5622 var: 0.0358
# Macro-F1_mean: 0.7236 var: 0.0193  Micro-F1_mean: 0.7245 var: 0.0186 auc 0.7421 var: 0.0187
# WeiboCOVID19
# Macro-F1_mean: 0.5582 var: 0.0505  Micro-F1_mean: 0.6030 var: 0.0580 auc 0.5864 var: 0.0576
# Macro-F1_mean: 0.6327 var: 0.0286  Micro-F1_mean: 0.6509 var: 0.0270 auc 0.6413 var: 0.0411
# PHEME
# Macro-F1_mean: 0.5656 var: 0.0473  Micro-F1_mean: 0.6111 var: 0.0424 auc 0.5666 var: 0.0357
# Macro-F1_mean: 0.6025 var: 0.0143  Micro-F1_mean: 0.6439 var: 0.0164 auc 0.5700 var: 0.0315
# TwitterCOVID19
# Macro-F1_mean: 0.6429 var: 0.0321  Micro-F1_mean: 0.6503 var: 0.0450 auc 0.6467 var: 0.0565
# Macro-F1_mean: 0.6537 var: 0.0321  Micro-F1_mean: 0.6745 var: 0.0298 auc 0.6366 var: 0.0234


# x*P
# DRWeiboV3
# Macro-F1_mean: 0.4954 var: 0.0295  Micro-F1_mean: 0.5023 var: 0.0237 auc 0.4880 var: 0.0241
# Macro-F1_mean: 0.6333 var: 0.0112  Micro-F1_mean: 0.6338 var: 0.0123 auc 0.6677 var: 0.0140
# Weibo
# Macro-F1_mean: 0.5271 var: 0.0633  Micro-F1_mean: 0.5590 var: 0.0478 auc 0.5958 var: 0.0694
# Macro-F1_mean: 0.7277 var: 0.0357  Micro-F1_mean: 0.7266 var: 0.0367 auc 0.7682 var: 0.0362
# WeiboCOVID19
# Macro-F1_mean: 0.5524 var: 0.0572  Micro-F1_mean: 0.6047 var: 0.0569 auc 0.5909 var: 0.0526
# Macro-F1_mean: 0.6530 var: 0.0312  Micro-F1_mean: 0.6668 var: 0.0465 auc 0.6699 var: 0.0461
# PHEME
# Macro-F1_mean: 0.5501 var: 0.0596  Micro-F1_mean: 0.5907 var: 0.0714 auc 0.5794 var: 0.0890
# Macro-F1_mean: 0.7056 var: 0.0073  Micro-F1_mean: 0.7425 var: 0.0088 auc 0.7633 var: 0.0086
# TwitterCOVID19
# Macro-F1_mean: 0.6405 var: 0.0166  Micro-F1_mean: 0.6738 var: 0.0161 auc 0.6344 var: 0.0158
# Macro-F1_mean: 0.6249 var: 0.0408  Micro-F1_mean: 0.6693 var: 0.0437 auc 0.6123 var: 0.0382

# x+P
# TwitterCOVID19
# Macro-F1_mean: 0.5924 var: 0.0566  Micro-F1_mean: 0.6268 var: 0.0444 auc 0.5914 var: 0.0460
# Macro-F1_mean: 0.6000 var: 0.0123  Micro-F1_mean: 0.6534 var: 0.0155 auc 0.5985 var: 0.0128
# PHEME
# Macro-F1_mean: 0.6040 var: 0.0306  Micro-F1_mean: 0.6360 var: 0.0453 auc 0.5856 var: 0.0497
# Macro-F1_mean: 0.6061 var: 0.0153  Micro-F1_mean: 0.6570 var: 0.0225 auc 0.5782 var: 0.0447
# WeiboCOVID19
# Macro-F1_mean: 0.5617 var: 0.0759  Micro-F1_mean: 0.5912 var: 0.0754 auc 0.5916 var: 0.0620
# Macro-F1_mean: 0.6704 var: 0.0311  Micro-F1_mean: 0.6837 var: 0.0221 auc 0.6592 var: 0.0365
# Weibo
# Macro-F1_mean: 0.5427 var: 0.0736  Micro-F1_mean: 0.5563 var: 0.0767 auc 0.5580 var: 0.0794
# Macro-F1_mean: 0.7056 var: 0.0175  Micro-F1_mean: 0.7066 var: 0.0177 auc 0.7092 var: 0.0186
# DRWeiboV3
# Macro-F1_mean: 0.5422 var: 0.0146  Micro-F1_mean: 0.5492 var: 0.0171 auc 0.5516 var: 0.0205
# Macro-F1_mean: 0.6176 var: 0.0108  Micro-F1_mean: 0.6187 var: 0.0106 auc 0.6361 var: 0.0134





# 目标数据DRWeiboV3
# no Weibo
# Macro-F1_mean: 0.5003 var: 0.0184  Micro-F1_mean: 0.5068 var: 0.0180 auc 0.5059 var: 0.0323
# Macro-F1_mean: 0.5244 var: 0.0142  Micro-F1_mean: 0.5363 var: 0.0134 auc 0.5209 var: 0.0173
# no Weibo W19
# Macro-F1_mean: 0.4919 var: 0.0177  Micro-F1_mean: 0.4994 var: 0.0164 auc 0.5042 var: 0.0126
# Macro-F1_mean: 0.5931 var: 0.0109  Micro-F1_mean: 0.5877 var: 0.0138 auc 0.6043 var: 0.0169
# no Weibo W19 PHEME
# Macro-F1_mean: 0.5105 var: 0.0181  Micro-F1_mean: 0.5209 var: 0.0147 auc 0.5186 var: 0.0191
# Macro-F1_mean: 0.5527 var: 0.0099  Micro-F1_mean: 0.5547 var: 0.0167 auc 0.5587 var: 0.0232


# Weibo
# no DRWeiboV3
# Macro-F1_mean: 0.5626 var: 0.0783  Micro-F1_mean: 0.5902 var: 0.0531 auc 0.6114 var: 0.0513
# Macro-F1_mean: 0.6733 var: 0.0142  Micro-F1_mean: 0.6749 var: 0.0142 auc 0.6942 var: 0.0147
# no DRWeiboV3 WeiboCOVID19
# Macro-F1_mean: 0.5372 var: 0.0325  Micro-F1_mean: 0.5459 var: 0.0254 auc 0.5567 var: 0.0378
# Macro-F1_mean: 0.6608 var: 0.0178  Micro-F1_mean: 0.6680 var: 0.0162 auc 0.6952 var: 0.0155
# no DRWeiboV3 WeiboCOVID19 PHEME
# Macro-F1_mean: 0.5449 var: 0.0453  Micro-F1_mean: 0.5598 var: 0.0360 auc 0.5660 var: 0.0429
# Macro-F1_mean: 0.7260 var: 0.0068  Micro-F1_mean: 0.7266 var: 0.0070 auc 0.7495 var: 0.0084


# 目标数据WeiboCOVID19
# no DRWeiboV3
# Macro-F1_mean: 0.4681 var: 0.1255  Micro-F1_mean: 0.5347 var: 0.1258 auc 0.6261 var: 0.0795
# Macro-F1_mean: 0.6589 var: 0.0299  Micro-F1_mean: 0.6837 var: 0.0356 auc 0.7120 var: 0.0283
# no DRWeiboV3 Weibo
# Macro-F1_mean: 0.4370 var: 0.0777  Micro-F1_mean: 0.4620 var: 0.0553 auc 0.5389 var: 0.0405
# Macro-F1_mean: 0.5819 var: 0.0204  Micro-F1_mean: 0.5927 var: 0.0246 auc 0.6200 var: 0.0209
# no DRWeiboV3 Weibo PHEME
# Macro-F1_mean: 0.4454 var: 0.1113  Micro-F1_mean: 0.5273 var: 0.1119 auc 0.5218 var: 0.0618
# Macro-F1_mean: 0.6296 var: 0.0301  Micro-F1_mean: 0.6654 var: 0.0162 auc 0.6429 var: 0.0495


# PHEME
# no DRWeiboV3
# Macro-F1_mean: 0.5054 var: 0.0427  Micro-F1_mean: 0.5237 var: 0.0633 auc 0.5325 var: 0.0555
# Macro-F1_mean: 0.5923 var: 0.0096  Micro-F1_mean: 0.6263 var: 0.0160 auc 0.5882 var: 0.0351


# MEAN 4.pt
# DRWeiboV3
# Macro-F1_mean: 0.5062 var: 0.0223  Micro-F1_mean: 0.5049 var: 0.0206 auc 0.4968 var: 0.0468
# Macro-F1_mean: 0.6405 var: 0.0090  Micro-F1_mean: 0.6384 var: 0.0150 auc 0.6984 var: 0.0132
# Weibo
# Macro-F1_mean: 0.5515 var: 0.0232  Micro-F1_mean: 0.5537 var: 0.0236 auc 0.5608 var: 0.0325
# Macro-F1_mean: 0.6407 var: 0.0099  Micro-F1_mean: 0.6413 var: 0.0104 auc 0.6499 var: 0.0098
# WeiboCOVID19
# Macro-F1_mean: 0.6405 var: 0.0707  Micro-F1_mean: 0.6643 var: 0.0877 auc 0.7256 var: 0.0719
# Macro-F1_mean: 0.7250 var: 0.0368  Micro-F1_mean: 0.7433 var: 0.0387 auc 0.8377 var: 0.0189
# PHEME
# Macro-F1_mean: 0.5807 var: 0.0205  Micro-F1_mean: 0.6173 var: 0.0280 auc 0.5854 var: 0.0486
# Macro-F1_mean: 0.5734 var: 0.0261  Micro-F1_mean: 0.6159 var: 0.0269 auc 0.5961 var: 0.0433
# TwitterCOVID19
# Macro-F1_mean: 0.6608 var: 0.0122  Micro-F1_mean: 0.6913 var: 0.0176 auc 0.7223 var: 0.0226
# Macro-F1_mean: 0.6337 var: 0.0244  Micro-F1_mean: 0.6631 var: 0.0303 auc 0.6964 var: 0.0326


# 都没有LN 5.pt
# DRWeiboV3
# Macro-F1_mean: 0.5262 var: 0.0359  Micro-F1_mean: 0.5282 var: 0.0330 auc 0.5325 var: 0.0422
# Macro-F1_mean: 0.6451 var: 0.0079  Micro-F1_mean: 0.6410 var: 0.0144 auc 0.6807 var: 0.0155
# Weibo
# Macro-F1_mean: 0.5431 var: 0.0496  Micro-F1_mean: 0.5563 var: 0.0405 auc 0.5754 var: 0.0538
# Macro-F1_mean: 0.7101 var: 0.0139  Micro-F1_mean: 0.7114 var: 0.0142 auc 0.7230 var: 0.0133
# WeiboCOVID19
# Macro-F1_mean: 0.5867 var: 0.0709  Micro-F1_mean: 0.6522 var: 0.0497 auc 0.6045 var: 0.0748
# Macro-F1_mean: 0.6597 var: 0.0289  Micro-F1_mean: 0.6751 var: 0.0303 auc 0.6836 var: 0.0420
# PHEME
# Macro-F1_mean: 0.5438 var: 0.0391  Micro-F1_mean: 0.5978 var: 0.0476 auc 0.5697 var: 0.0506
# Macro-F1_mean: 0.6067 var: 0.0111  Micro-F1_mean: 0.6555 var: 0.0116 auc 0.6525 var: 0.0182
# TwitterCOVID19
# Macro-F1_mean: 0.6431 var: 0.0335  Micro-F1_mean: 0.6691 var: 0.0409 auc 0.6642 var: 0.0412
# Macro-F1_mean: 0.6563 var: 0.0266  Micro-F1_mean: 0.6876 var: 0.0261 auc 0.6337 var: 0.0437



# Weibo
# Macro-F1_mean: 0.5450 var: 0.0812  Micro-F1_mean: 0.5626 var: 0.0627 auc 0.5889 var: 0.0674
# Macro-F1_mean: 0.7201 var: 0.0250  Micro-F1_mean: 0.7200 var: 0.0236 auc 0.7469 var: 0.0205
# Macro-F1_mean: 0.7298 var: 0.0100  Micro-F1_mean: 0.7293 var: 0.0111 auc 0.7476 var: 0.0158
# Macro-F1_mean: 0.7632 var: 0.0132  Micro-F1_mean: 0.7597 var: 0.0119 auc 0.7912 var: 0.0180
# WeiboCOVID19
# Macro-F1_mean: 0.5519 var: 0.0864  Micro-F1_mean: 0.6455 var: 0.0302 auc 0.5936 var: 0.0773
# Macro-F1_mean: 0.6873 var: 0.0347  Micro-F1_mean: 0.6986 var: 0.0370 auc 0.7177 var: 0.0298
# Macro-F1_mean: 0.7141 var: 0.0156  Micro-F1_mean: 0.7348 var: 0.0207 auc 0.7603 var: 0.0259
# Macro-F1_mean: 0.7190 var: 0.0195  Micro-F1_mean: 0.7382 var: 0.0234 auc 0.7422 var: 0.0382
# TwitterCOVID19
# Macro-F1_mean: 0.5911 var: 0.0589  Micro-F1_mean: 0.6396 var: 0.0754 auc 0.5917 var: 0.0523
# Macro-F1_mean: 0.6066 var: 0.0358  Micro-F1_mean: 0.6376 var: 0.0421 auc 0.5805 var: 0.0432
# Macro-F1_mean: 0.5941 var: 0.0196  Micro-F1_mean: 0.6482 var: 0.0341 auc 0.5999 var: 0.0200
# Macro-F1_mean: 0.6584 var: 0.0149  Micro-F1_mean: 0.6996 var: 0.0160 auc 0.6587 var: 0.0141


# DRWeiboV3
# 0.1
# Macro-F1_mean: 0.6193 var: 0.0083  Micro-F1_mean: 0.6199 var: 0.0099 auc 0.6571 var: 0.0120
# 0.3
# Macro-F1_mean: 0.6560 var: 0.0097  Micro-F1_mean: 0.6483 var: 0.0173 auc 0.6911 var: 0.0245
# 0.5
# Macro-F1_mean: 0.6281 var: 0.0083  Micro-F1_mean: 0.6291 var: 0.0085 auc 0.6694 var: 0.0145
# 0.7
# Macro-F1_mean: 0.5391 var: 0.0139  Micro-F1_mean: 0.5270 var: 0.0197 auc 0.5148 var: 0.0462
# 0.9
# Macro-F1_mean: 0.6291 var: 0.0079  Micro-F1_mean: 0.6232 var: 0.0110 auc 0.6590 var: 0.0156

# PHEME
# 0.1
# Macro-F1_mean: 0.5747 var: 0.0122  Micro-F1_mean: 0.6265 var: 0.0108 auc 0.5849 var: 0.0366
# 0.3
# Macro-F1_mean: 0.5998 var: 0.0145  Micro-F1_mean: 0.6394 var: 0.0099 auc 0.6140 var: 0.0423
# 0.5
# Macro-F1_mean: 0.5682 var: 0.0254  Micro-F1_mean: 0.6151 var: 0.0252 auc 0.6003 var: 0.0334
# 0.7
# Macro-F1_mean: 0.6216 var: 0.0207  Micro-F1_mean: 0.6688 var: 0.0156 auc 0.6973 var: 0.0227
# 0.9
# Macro-F1_mean: 0.6497 var: 0.0325  Micro-F1_mean: 0.6721 var: 0.0289 auc 0.7197 var: 0.0338
