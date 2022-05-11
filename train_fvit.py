import torch
from module.ViT import SharedFViT, TwoFViT
import torch.optim as optim
import torch.nn as nn
import argparse
import yaml
from time import time
import glob
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset.vins_dataset import VinsDataset
from tensorboardX import SummaryWriter
from losses.loss_functions import Shrinkage_loss


def main(configs):
    _debug = False
    # log writer
    writer = SummaryWriter()
    # model and data configuration
    m_configs = yaml.safe_load(open(configs.model_cfg, 'r'))
    model_cfg = m_configs['model']
    data_cfg = m_configs['data']
    # hyper parameters
    learning_rate = float(model_cfg['learning_rate'])
    weight_decay = float(model_cfg['weight_decay'])
    intermediate_channels = list(model_cfg['intermediate_channels'])
    num_patches = int(model_cfg['num_patches'])
    patch_size = int(model_cfg['patch_size'])
    pos_dim = int(model_cfg['pos_dim'])
    emb_dim = int(model_cfg['emb_dim'])
    code_dim = int(model_cfg['code_dim'])
    depth = int(model_cfg['depth'])
    heads = int(model_cfg['heads'])
    mlp_dim = int(model_cfg['mlp_dim'])
    pool = model_cfg['pool']
    channels = int(model_cfg['channels'])
    dim_head = int(model_cfg['dim_head'])
    dropout = float(model_cfg['dropout'])
    emb_dropout = float(model_cfg['emb_dropout'])
    batch_size = int(model_cfg['batch_size'])
    max_epochs = int(model_cfg['max_epochs'])
    epsilon_w = float(model_cfg['epsilon_w'])
    momentum = float(model_cfg['momentum'])
    scheduler_gamma = float(model_cfg['scheduler_gamma'])
    shrinkage_a = float(model_cfg['shrinkage_a'])
    shrinkage_c = float(model_cfg['shrinkage_c'])
    # device of model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # initialize model
    model = SharedFViT(
        num_patches=num_patches,
        patch_size=patch_size,
        pos_dim=pos_dim,
        emb_dim=emb_dim,
        code_dim=code_dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        pool=pool,
        channels=channels,  # rgbd
        dim_head=dim_head,
        dropout=dropout,
        emb_dropout=emb_dropout
    )

    # loss function define
    criterion = torch.nn.MSELoss().to(device)
    # criterion = Shrinkage_loss(shrinkage_a, shrinkage_c).to(device)
    # optimizer define
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, scheduler_gamma)
    model.to(device)
    criterion.to(device)
    # prepare the data
    """ dummy test data
    img = torch.randn(1, 128, 16, 16, 4).to(device)
    pos = torch.randn(1, 128, 3).to(device)
    img2 = torch.randn(1, 128, 16, 16, 4).to(device)
    pos2 = torch.randn(1, 128, 3).to(device)
    preds = model(img, pos, img2, pos2).to(device)  # (1, 1000)
    gt_score = torch.tensor([[0.01]]).to(device)
    print(preds.shape)
    """
    # training dataset and validation dataset
    train_data_root = data_cfg['train_dataset']
    val_data_root = data_cfg['val_dataset']
    ckpt_out_path = data_cfg['ckpt_output']
    summary_out_path = data_cfg['summary_output']
    train_dataset = VinsDataset(train_data_root, split='train')
    val_dataset = VinsDataset(val_data_root, split='val')
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8
    )
    best_model_loss = 1e5
    for epoch in range(max_epochs):
        train_iter = iter(train_dataloader)
        val_iter = iter(val_dataloader)
        # Training
        if config.continue_training:
            checkpoint_fnames = sorted(list(glob.glob(os.path.join(ckpt_out_path, "*.pth"))))
            if len(checkpoint_fnames) > 0:
                latest_ckpt_path = checkpoint_fnames[-1]
                model_dict = torch.load(latest_ckpt_path)
                model.load_state_dict(model_dict["state_dict"])
                optimizer.load_state_dict(model_dict['optimizer'])
                scheduler.load_state_dict(model_dict['scheduler'])
            else:
                print("path: {} has no checkpoint file".format(ckpt_out_path))
        model.train()
        bar = tqdm(desc="Training Epoch:{}/{}".format(epoch, max_epochs - 1), initial=0,
                   total=len(train_dataloader), unit='batches', dynamic_ncols=True, bar_format="{l_bar}{bar:12}{r_bar}")
        for i, data in enumerate(train_iter):
            if _debug and i > 5:
                print('------debug mode on---------')
                break
            tstart = time()
            optimizer.zero_grad()
            patch_ref = data['patch_data_ref'].float().to(device)
            patch_next = data['patch_data_next'].float().to(device)
            pos_ref = data['key_points_xyz_data_ref'].float().to(device)
            pos_next = data['key_points_xyz_data_next'].float().to(device)
            gt_score = data['iou_data'].float().to(device)
            preds = model(patch_ref, pos_ref, patch_next, pos_next).squeeze()  # (B, 1)
            loss = criterion(preds, gt_score)
            loss.backward()
            optimizer.step()
            scheduler.step()
            time_spend = time() - tstart
            with torch.no_grad():
                bar.update(1)
                bar_dict = {}
                bar_dict['loss'] = loss.cpu().numpy()
                for ll, vv in bar_dict.items():
                    if isinstance(vv, str):
                        continue
                    bar_dict[ll] = round(float(vv), 3)
                bar.set_postfix(bar_dict)
                writer.add_scalar('train/loss', loss.cpu().numpy(), i)
                writer.flush()
        # Evaluation
        bar.close()
        model.eval()
        batches = len(val_dataloader)
        with torch.no_grad():
            valid_bar = tqdm(desc="Validating", initial=0, total=batches, unit='batches', dynamic_ncols=True,
                             bar_format="{l_bar}{bar:12}{r_bar}")
            val_acc_loss = 0
            for i, data in enumerate(val_iter):
                tstart = time()
                patch_ref = data['patch_data_ref'].float().to(device)
                patch_next = data['patch_data_next'].float().to(device)
                pos_ref = data['key_points_xyz_data_ref'].float().to(device)
                pos_next = data['key_points_xyz_data_next'].float().to(device)
                gt_score = data['iou_data'].float().to(device)
                preds = model(patch_ref, pos_ref, patch_next, pos_next).squeeze()  # (B, 1)
                loss = criterion(preds, gt_score)
                writer.add_scalar('val/loss', loss.cpu().numpy(), i)
                val_acc_loss += loss.cpu().numpy()
                val_bar_dict = {}
                val_bar_dict['loss'] = loss.cpu().numpy()
                for ll, vv in val_bar_dict.items():
                    if isinstance(vv, str):
                        continue
                    val_bar_dict[ll] = round(float(vv), 3)
                valid_bar.set_postfix(val_bar_dict)
                valid_bar.update(i)

            val_acc_loss = val_acc_loss / len(val_dataset)
            if val_acc_loss < best_model_loss:
                torch.save(
                    {
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'curr_epoch': epoch,
                    }, os.path.join(ckpt_out_path, "weights_epoch_best.pth"))

        torch.save(
            {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'curr_epoch': epoch,
            }, os.path.join(ckpt_out_path, "weights_epoch_{}.pth".format(epoch)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_name', type=str, default="weights_epoch_3.pth")
    parser.add_argument('--continue_training', type=bool, default=False)
    parser.add_argument(
        '--model_cfg', '-dc',
        type=str,
        required=False,
        default='config/vit_config.yaml',
        help='Classification yaml cfg file. See /config/labels for sample. No default!',
    )
    config = parser.parse_args()
    main(config)