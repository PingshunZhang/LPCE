import time
import numpy as np
import torch

from utils.utils import compute_metrics

def eval_model(model, dataloader, model_path=None):
    print("Start evaluation...")
    since = time.time()

    print("Loading model parameters from {}".format(model_path))
    model.load_state_dict(torch.load(model_path))

    was_training = model.training
    model.eval()
    model.feat_ext.feat_ext1.eval()
    model.feat_ext.feat_ext2.eval()

    ds = dataloader.dataset
    classes = ds.classes
    cls_cache = ds.cls

    all_pck = []
    all_iou = []
    all_lv = []
    all_overlap = []
    category_pck = np.zeros(len(classes))
    category_iou = np.zeros(len(classes))
    category_lv = np.zeros(len(classes))
    category_overlap = np.zeros(len(classes))
    test_time = []

    for i, cls in enumerate(classes):
        iter_num = 0

        ds.set_cls(cls)
        for inputs in dataloader:
            input_graphs = inputs['graphs'].to('cuda')
            im_sizes = inputs['im_sizes'].to('cuda')
            L_pcks = inputs['L_pcks'].to('cuda')
            images = inputs['images'].to('cuda')
            file_name = inputs['file_name']

            iter_num = iter_num + 1

            with torch.set_grad_enabled(False):
                start = time.time()
                output_graphs = model(input_graphs, images, file_name)

                test_time += [time.time() - start]

            output_graphs_list = output_graphs.to_data_list()
            batch_pck, batch_iou, batch_lv, batch_overlap = compute_metrics(output_graphs_list, im_sizes, L_pcks)

            all_pck += batch_pck
            category_pck[i] += batch_pck[0]
            all_iou += batch_iou
            category_iou[i] += batch_iou[0]
            all_lv += batch_lv
            category_lv[i] += batch_lv[0]
            all_overlap += batch_overlap
            category_overlap[i] += batch_overlap[0]

        category_pck[i] /= len(ds)
        category_iou[i] /= len(ds)
        category_lv[i] /= len(ds)
        category_overlap[i] /= len(ds)

    pck_avg = np.mean(np.array(all_pck))
    iou_avg = np.mean(np.array(all_iou))
    lv_avg = np.mean(np.array(all_lv))
    overlap_avg = np.mean(np.array(all_overlap))
    time_avg = np.mean(np.array(test_time))
    time_std = np.std(np.array(test_time))

    print("Accuracy")
    for i, cls in enumerate(classes):
        print('Class {:<20}  PCK = {:.4f}  IOU = {:.4f}  LV = {:.4f}  Overlap = {:.4f}'.format(cls, category_pck[i], category_iou[i], category_lv[i], category_overlap[i]))
    print("Mean PCK = {:.4f}  IOU = {:.4f}  LV = {:.4f}  Overlap = {:.4f}".format(pck_avg, iou_avg, lv_avg, overlap_avg))
    print("Mean generating time = {:.6f}s, std = {:.6f}".format(time_avg, time_std))


    time_elapsed = time.time() - since
    print("Evaluation complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print()


    ds.cls = cls_cache