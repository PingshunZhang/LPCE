import torch
from pathlib import Path
import numpy as np
import random

from data.data_loader_graph_Full_gussian import GMDataset, get_dataloader

from eval_fn_style import eval_model
from utils.config import cfg
from modules.LPModel_style import LPModel
from utils.utils import update_params_from_cmdline

## ***********************************************************************
## Testing
## **********************************************************************
def seed_torch(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
     
if __name__ == "__main__":
    from utils.dup_stdout_manager import DupStdoutFileManager

    cfg = update_params_from_cmdline(default_params=cfg)
    import json
    import os

    model_dir = 'trained_style_models'
    model_name = 'params.pt'

    with open(os.path.join(model_dir, "test_settings.json"), "w") as f:
        json.dump(cfg, f)

    with DupStdoutFileManager(str(Path(model_dir) / ("test_log.log"))) as _:
        seed_torch(cfg.RANDOM_SEED + 1)

        graph_dataset = GMDataset(cfg.DATASET_NAME, sets='test', length=None, img_resize=(256, 256))
        dataloader = get_dataloader(graph_dataset, batch_size=cfg.EVAL.BATCH_SIZE, fix_seed=True, shuffle=False)

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
                        train_fe=False,
                        normalization=True,
                        backbone='resnet101')
        model = model.cuda()

        eval_pck = eval_model(model, dataloader, model_path=model_dir + '/' + model_name)
