import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import numpy as np
from tqdm import tqdm
from arguments import get_args
from augmentations import get_aug
from models import get_model
from tools import AverageMeter, knn_monitor, Logger, file_exist_check
from datasets import get_dataset, BoneXray1st, LumbarBoneXray1st
from optimizers import get_optimizer, LR_Scheduler
from linear_eval import main as linear_eval
from datetime import datetime
# from datasets.BoneXray1st import TrainingDataset
from datasets.LumbarBoneXray1st import TrainingDataset


def main(device, args):
    dataset_train = TrainingDataset(split_fold=0,
                                    # image_size=args.dataset.image_size, load_size=args.dataset.load_size,
                                    aug_conf="paired_synthesis", n_worker=args.dataset.num_workers, mode='train')

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.train.batch_size,
        shuffle=True,
        **args.dataloader_kwargs)

    # define model
    model = get_model(args.model).to(device)
    # model = torch.nn.DataParallel(model)

    # define optimizer
    optimizer = get_optimizer(
        args.train.optimizer.name, model, 
        lr=args.train.base_lr*args.train.batch_size/256, 
        momentum=args.train.optimizer.momentum,
        weight_decay=args.train.optimizer.weight_decay)

    lr_scheduler = LR_Scheduler(
        optimizer,
        args.train.warmup_epochs, args.train.warmup_lr*args.train.batch_size/256, 
        args.train.num_epochs, args.train.base_lr*args.train.batch_size/256, args.train.final_lr*args.train.batch_size/256, 
        len(train_loader),
        constant_predictor_lr=True # see the end of section 4.2 predictor
    )

    logger = Logger(tensorboard=args.logger.tensorboard, matplotlib=args.logger.matplotlib, log_dir=os.path.join('workspace', args.name, args.log_dir))
    accuracy = 0.
    # Start training
    global_progress = tqdm(range(0, args.train.stop_at_epoch), desc=f'Training')
    for epoch in global_progress:
        model.train()
        loss_epoch = 0.
        B = 0.
        local_progress=tqdm(train_loader, desc=f'Epoch {epoch}/{args.train.num_epochs}', disable=args.hide_progress)
        for idx, (images) in enumerate(local_progress):

            model.zero_grad()
            data_dict = model.forward(images[0].to(device, non_blocking=True), images[1].to(device, non_blocking=True))
            loss = data_dict['loss'].mean() # ddp
            loss_epoch += loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            data_dict.update({'lr':lr_scheduler.get_lr()})
            
            local_progress.set_postfix(data_dict)
            logger.update_scalers(data_dict)

            B += 1


        epoch_dict = {"epoch":epoch + 1, "loss":loss_epoch / B}
        global_progress.set_postfix(epoch_dict)
        logger.update_scalers(epoch_dict)
    
    # Save checkpoint
        if (epoch + 1) % 100 == 0:
            save_path = os.path.join('workspace', args.name, args.ckpt_dir)
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.backbone.encoder.state_dict(), os.path.join(save_path, f'ckp_netG_enc_{epoch + 1}.pt'))
            torch.save(model.backbone.fuse.state_dict(),
                       os.path.join(save_path, f'ckp_netG_fus_{epoch + 1}.pt'))
            print(f"Model saved {epoch + 1}")

    # if args.eval is not False:
    #     args.eval_from = model_path
    #     linear_eval(args)


if __name__ == "__main__":
    args = get_args()
    logs_save_path = os.path.join('workspace', args.name, args.log_dir)
    os.makedirs(logs_save_path, exist_ok=True)
    main(device=args.device, args=args)

    # completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')
    #
    #
    #
    # os.rename(args.log_dir, completed_log_dir)
    # print(f'Log file has been saved to {completed_log_dir}')














