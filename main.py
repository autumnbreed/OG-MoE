import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from transformers import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.activations import ACT2FN

from data_loader import create_dataloader, load_adni_img_gene_data_
from model import MiniViT, DemoConfig, train, evaluate
from utils import Logger, save_checkpoint, load_checkpoint

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(config):
    # seed
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) model
    model = MiniViT(config).to(device)

    # 3) data
    ######################### Replace with local data path #################################
    training_data = load_adni_img_gene_data_('.../Data/train')
    train_loader = create_dataloader(training_data, batch_size=config.batch_size)
    
    ########################## Replace with local data path #################################
    test_data = load_adni_img_gene_data_('.../Data/test') 
    test_loader = create_dataloader(test_data, batch_size=100)

    # 4)  Logger
    logger = Logger(log_dir=config.log_dir, log_txt=config.log_txt)

    # 5) Warm Start/start_from_checkpoint
    # if config.start_from_checkpoint:
    #     start_global_step = load_checkpoint(model, None, config.start_from_checkpoint, device=device)
    #     print(f"Continue from step {start_global_step}")
    # else:
    #     print("=== Warm Start Phase ===")
    #     warm_optimizer = AdamW(model.parameters(), lr=config.warm_start_lr)
    #     model.train()
    #     global_step = 0
    #     while global_step < config.warm_start_steps:
    #         for batch in train_loader:
    #             global_step += 1
    #             patch_tensors = batch["patch_tensors"].to(device)
    #             # warm_start = True
    #             outputs = model(
    #                 patch_tensors,
    #                 warm_start=True,
    #                 mask_ratio=config.mask_ratio
    #             )
    #             loss = outputs.loss

    #             warm_optimizer.zero_grad()
    #             loss.backward()
    #             warm_optimizer.step()

    #             if global_step % 50 == 0:
    #                 logger.log(global_step, {"warm_loss": loss.item()})

    #             if global_step % config.save_steps == 0:
    #                 ckpt_path = os.path.join(config.output_dir, f"warm_ckpt_step_{global_step}.pt")
    #                 save_checkpoint(model, warm_optimizer, global_step, ckpt_path)

    #             if global_step >= config.warm_start_steps:
    #                 break

        # # save warm checkpoint
        # warm_ckpt = os.path.join(config.output_dir, f"warm_final.pt")
        # save_checkpoint(model, warm_optimizer, global_step, warm_ckpt)
        # print(f"Warm Start done. Saved warm checkpoint: {warm_ckpt}")

        # # warm checkpoint 
        # load_checkpoint(model, None, warm_ckpt, device=device)
        # start_global_step = 0

    # 6) Traing (CLS + MSE)
    print("=== Main Training Phase ===")
    optimizer = AdamW(model.parameters(), lr=config.train_lr)
    #scheduler = StepLR(optimizer, step_size=config.stepsize, gamma=config.gamma)
    scheduler = MultiStepLR(optimizer, milestones=[5, 10, 40, 80], gamma=0.1,)

    # model.train()

    #criterion
    global_step = 0
    for epoch in range(config.epochs):
        ep_loss, rec_loss, global_step, ep_train_acc, f1_dicts= train(model, 
                                                                train_loader,
                                                                device,
                                                                global_step,
                                                                logger,
                                                                optimizer,
                                                                patch_mask=True,
                                                                masked_rate=config.mask_ratio,
                                                                rec_percent=config.rec_percent
                                                                )
        scheduler.step()
        # logger.log(epoch, {
        #             "train_loss": loss.item(),
        #             "acc1": acc1,
        #             "lr": optimizer.param_groups[0]["lr"]
        #         })
        logger.log(global_step, {
                    "train_loss": ep_loss,
                    "recMSE":rec_loss,
                    "acc_train": ep_train_acc,
                    "lr": optimizer.param_groups[0]["lr"],
                    "train_F1":f1_dicts.__repr__()
                })
        
        ev_acc, ev_f1 = evaluate(model, 
            test_loader,
            device,
            global_step,
            logger,)
        
        # logger.log(global_step, {
        #             "acc_test": ev_acc,
        #             "test_F1":ev_f1.__repr__()
        #         })
        logger.log(global_step, {
                    "acc_test": ev_acc,
                    "test_F1":ev_f1
                })
        # print('-------------: Epoch {}, Step {}, Avg ACC {}.'.format(epoch, global_step, ev_acc))
        print('-------------: Epoch {}, Step {}.'.format(epoch, global_step))
        print('-- Train loss {}, RecMSE {}, train_F1{}'.format(ep_loss, rec_loss, f1_dicts.__repr__()))
        print('-- Test', ev_acc, ' F1 ', ev_f1)
        
        if epoch % config.save_steps == 0:
            ckpt_path = os.path.join(config.output_dir, f"ckpt_step_{global_step}.pt")
            save_checkpoint(model, optimizer, global_step, ckpt_path)

    # save
    model.save_pretrained(config.saves)

    # # load
    # new_model = MyModel.from_pretrained("my_model_path")
        
    logger.close()
    print("Training complete.")

if __name__ == "__main__":
    import sys
    config_file = sys.argv[1]
    with open(config_file, "r") as f:
        config_dict = json.load(f)

    config = DemoConfig(**config_dict)
    
    main(config)
