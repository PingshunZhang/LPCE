import time
from pathlib import Path
import numpy as np
import torch
import os

output_dir = 'data/downloaded/one_style_feature'


def eval_model(model, dataloader, model_path=None):
    print("Start evaluation...")

    print("Loading model parameters from {}".format(model_path))
    model.load_state_dict(torch.load(model_path))

    was_training = model.training
    model.eval()

    ds = dataloader.dataset

    classes = ds.classes
    cls_cache = ds.cls

    all_correct = []
    for i, cls in enumerate(classes):

        iter_num = 0

        ds.set_cls(cls)
        for inputs in dataloader:
            input_graphs = inputs['graphs'].to('cuda')
            style = inputs['style'].to('cuda')
            images = inputs['images'].to('cuda')
            file_name = inputs['file_name']
            iter_num = iter_num + 1

            with torch.set_grad_enabled(False):
                output_graphs, style_features= model(input_graphs, images)
            torch.save(style_features, os.path.join(output_dir, f"{file_name[0]}.pt"))

            pred = output_graphs.z.argmax(dim=-1)
            batch_correct = (pred == style).sum().item()
            all_correct += [batch_correct]

    acc_avg = np.mean(np.array(all_correct))
    print(acc_avg)

    ds.cls = cls_cache

