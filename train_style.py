import torch
import torch.optim as optim
import time
from pathlib import Path
import numpy as np

from data.data_loader_graph_Full_gussian import GMDataset, get_dataloader

from modules.loss import focal_loss
from utils.config import cfg

from modules.LPModel_style import LPModel

from utils.utils import delete_folder_contents_except, seed_torch, update_params_from_cmdline


lr_schedules = {
    "long_halving": (80, (16, 32, 48, 64, 80), 0.5)
}

def train_eval_model(model, criterion, optimizer, dataloader, num_epochs, start_epoch=0, exp_id=0):
    print("Start training...")

    dataset_size = len(dataloader["train"].dataset)

    device = next(model.parameters()).device
    print("model on device: {}".format(device))

    checkpoint_path = Path(cfg.model_dir) / str(exp_id + 1) / "trained_style_models"
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True)

    _, lr_milestones, lr_decay = lr_schedules[cfg.TRAIN.lr_schedule]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_decay)

    loss_table = np.zeros((num_epochs, 1))

    for epoch in range(start_epoch, num_epochs):
        print("-" * 10)
        print("Epoch {}/{}".format(epoch + 1, num_epochs))

        model.train()  # Set model to training mode
        model.feat_ext.feat_ext1.train()
        model.feat_ext.feat_ext2.train()

        print("lr = " + ", ".join(["{:.2e}".format(x["lr"]) for x in optimizer.param_groups]))

        epoch_loss = 0.0
        running_loss = 0.0
        iter_num = 0

        # Iterate over data.
        for inputs in dataloader["train"]:
            input_graphs = inputs['graphs'].to('cuda')
            style = inputs['style'].to('cuda')
            images = inputs['images'].to('cuda')

            iter_num = iter_num + 1

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                # forward
                output_graphs, style_features = model(input_graphs, images)

                loss = criterion(output_graphs.z, style)

                # backward + optimize
                loss.backward()
                optimizer.step()

                # statistics
                bs = inputs['n_nodes'].size(0)
                running_loss += loss.item() * bs
                epoch_loss += loss.item() * bs

                if iter_num % cfg.TRAIN.STATISTIC_STEP == 0:
                    loss_avg = running_loss / cfg.TRAIN.STATISTIC_STEP / bs
                    print(
                        "Epoch {:<4} Iter {:<4}  Loss={:<8.4f}".format(
                            epoch + 1, iter_num, loss_avg
                        )
                    )

                    running_loss = 0.0

        epoch_loss = epoch_loss / dataset_size
        loss_table[epoch] = epoch_loss

        if cfg.save_checkpoint:
            base_path = Path(checkpoint_path / "{:04}".format(epoch + 1))
            Path(base_path).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(base_path / "trained_style_models.pt"))

        print("Over whole epoch {:<4} -------- Loss: {:.4f}".format(epoch + 1, epoch_loss))
        print()

        scheduler.step()

    best_loss_epoch = np.argmin(loss_table) + 1


    print('minimum loss {:.4f} at epoch {}/{}'.
          format(np.min(loss_table), best_loss_epoch, num_epochs))
    print()

    excluded_folder = ["{:04}".format(best_loss_epoch), "{:04}".format(num_epochs)]
    delete_folder_contents_except(checkpoint_path, excluded_folder)

    return best_loss_epoch


if __name__ == "__main__":
    from utils.dup_stdout_manager import DupStdoutFileManager

    cfg = update_params_from_cmdline(default_params=cfg)
    import json
    import os

    os.makedirs(cfg.model_dir, exist_ok=True)
    with open(os.path.join(cfg.model_dir, "settings.json"), "w") as f:
        json.dump(cfg, f)

    with DupStdoutFileManager(str(Path(cfg.model_dir) / ("train_log.log"))) as _:
        n_exp = 2
        start_time = time.time()
        for exp_id in range(n_exp):
            seed_torch(cfg.RANDOM_SEED + 1)
            dataset_bs = {'train': cfg.TRAIN.BATCH_SIZE, 'test': cfg.EVAL.BATCH_SIZE}
            graph_dataset = {x: GMDataset(cfg.DATASET_NAME, sets=x, length=None, img_resize=(256, 256)) for x in ("train", "test")}
            dataloader = {x: get_dataloader(graph_dataset[x], dataset_bs[x],
                                            fix_seed=(x == "test"), shuffle=(x == "train")) for x in ("train", "test")}

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model = LPModel(n_layers=3,
                            n_heads=16,
                            node_input_dim=4,
                            edge_input_dim=6,
                            node_dim=288,
                            edge_dim=288,
                            node_hid_dim=96,
                            edge_hid_dim=48,
                            output_dim=2,
                            disable_edge_updates=False,
                            train_fe=True,
                            # train_fe=False,
                            normalization=True,
                            backbone='resnet101')
            model = model.cuda()

            lr_init = [0.0001]
            for param in model.parameters():
                param.requires_grad = False

            for param in model.feat_ext.feat_ext1.parameters():
                param.requires_grad = True
            for param in model.feat_ext.feat_ext2.parameters():
                param.requires_grad = True

            for param in model.lp_gnns.parameters():
                param.requires_grad = True

                criterion = focal_loss(alpha=[0.2183,0.2187,0.1112,0.2115,0.2403], gamma=2, num_classes = 5)

                optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                        lr=lr_init[0],
                                        betas=(0.9, 0.999),
                                        eps=1e-08,
                                        weight_decay=0.01)

                num_epochs, _, __ = lr_schedules[cfg.TRAIN.lr_schedule]
                best_epoch = train_eval_model(model,
                                            criterion,
                                            optimizer,
                                            dataloader,
                                            num_epochs=num_epochs,
                                            start_epoch=0,
                                            exp_id=exp_id)

        total_time = time.time() - start_time
        print('Total training & evaluation time: {:.0f}h {:.0f}m {:.0f}s'
              .format(total_time // 3600, (total_time // 60) % 60, total_time % 60))