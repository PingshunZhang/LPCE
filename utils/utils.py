import ast
import collections
import json
import os
import sys
import time
from collections.abc import Mapping
from copy import deepcopy
from warnings import warn
from torchvision.ops import box_iou
import shutil
import random
import numpy as np

import torch
JSON_FILE_KEY = 'default_json'

def delete_folder_contents_except(folder_path, excluded_folder):
    if os.path.exists(folder_path):
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                if item in excluded_folder:
                    continue
                shutil.rmtree(item_path)

def seed_torch(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def unnormalize_points(points_norm, im_size):
  w, h = im_size[0], im_size[1]
  points = points_norm.clone()
  points[:, 0] *= w
  points[:, 1] *= h

  return points

def PCK(p_src, p_wrp, L_pck, alpha=0.05):

  point_distance = torch.pow(torch.sum(torch.pow(p_src - p_wrp, 2), 1), 0.5)
  L_pck_mat = L_pck.expand_as(point_distance)
  correct_points = torch.le(point_distance, L_pck_mat * alpha)
  pck = torch.mean(correct_points.float())
  return pck

def IOU(points_gt, points_pred, label_size):
  points_gt_1 = points_gt - label_size/2
  points_gt_2 = points_gt + label_size/2
  box_gt = torch.concatenate((points_gt_1,points_gt_2), dim=-1)
  points_pred_1 = points_pred - label_size/2
  points_pred_2 = points_pred + label_size/2
  box_pred = torch.concatenate((points_pred_1,points_pred_2), dim=-1)
  iou = []
  for i in range(box_gt.shape[0]):
    iou.append(box_iou(box_gt[i,:].unsqueeze(0), box_pred[i,:].unsqueeze(0)))
  return torch.cat(iou).mean()

def Overlap(points_pred, label_size):
  points_pred_1 = points_pred - label_size/2
  points_pred_2 = points_pred + label_size/2
  box_pred = torch.concatenate((points_pred_1,points_pred_2), dim=-1)
  if box_pred.shape[0] == 1:
     label_iou = torch.tensor(0, device='cuda:0')
  else:
    iou = []
    for i in range(box_pred.shape[0]):
      for j in range(i+1, box_pred.shape[0]):
        iou.append(box_iou(box_pred[i].unsqueeze(0),box_pred[j].unsqueeze(0)))
    label_iou = torch.cat(iou).mean()
  return label_iou

def LV(points_pred, label_size):
  points_pred_1 = points_pred - label_size/2
  points_pred_2 = points_pred + label_size/2
  box_pred = torch.concatenate((points_pred_1,points_pred_2), dim=-1)
  if box_pred.shape[0] == 1:
     lv_mean = torch.tensor(1, device='cuda:0')
  else:
    lvs = []
    for i in range(box_pred.shape[0]):
      lv = torch.tensor(1.00, device='cuda:0').unsqueeze(0).unsqueeze(0)
      for j in range(box_pred.shape[0]):
        if i != j:
           current_iou = box_iou(box_pred[i].unsqueeze(0),box_pred[j].unsqueeze(0))
           if current_iou > 0:
            lv = torch.tensor(0.00, device='cuda:0').unsqueeze(0).unsqueeze(0)
            break
      lvs.append(lv)
    lv_mean = torch.cat(lvs).mean()
  return lv_mean

def compute_pck(graphs, im_sizes, L_pcks):
  pck = []
  alpha = 0.05
  for i in range(len(graphs)):
    points_norm_pred = graphs[i].x + graphs[i].y[:, 2:4]
    points_norm_gt = graphs[i].y[:, :2] + graphs[i].y[:, 2:4]
    im_size = im_sizes[i]
    points_pred = unnormalize_points(points_norm_pred, im_size)
    points_gt = unnormalize_points(points_norm_gt, im_size)
    pck.append(PCK(points_gt, points_pred, L_pcks[i], alpha).cpu().numpy())

  return pck

def compute_metrics(graphs, im_sizes, L_pcks):
  pck = []
  iou = []
  lv = []
  overlap = []
  alpha = 0.05
  for i in range(len(graphs)):
    points_norm_pred = graphs[i].x + graphs[i].y[:, 2:4]
    points_norm_gt = graphs[i].y[:, :2] + graphs[i].y[:, 2:4]
    im_size = im_sizes[i]
    points_pred = unnormalize_points(points_norm_pred, im_size)
    points_gt = unnormalize_points(points_norm_gt, im_size)
    pck.append(PCK(points_gt, points_pred, L_pcks[i], alpha).cpu().numpy())

    label_size = unnormalize_points(graphs[i].y[:, 4:6], im_size)
    iou.append(IOU(points_gt, points_pred, label_size).cpu().numpy())
    lv.append(LV(points_pred, label_size).cpu().numpy())
    overlap.append(Overlap(points_pred, label_size).cpu().numpy())

  return pck, iou, lv, overlap

class ParamDict(dict):
  """ An immutable dict where elements can be accessed with a dot"""
  __getattr__ = dict.__getitem__

  def __delattr__(self, item):
    raise TypeError("Setting object not mutable after settings are fixed!")

  def __setattr__(self, key, value):
    raise TypeError("Setting object not mutable after settings are fixed!")

  def __setitem__(self, key, value):
    raise TypeError("Setting object not mutable after settings are fixed!")

  def __deepcopy__(self, memo):
    """ In order to support deepcopy"""
    return ParamDict([(deepcopy(k, memo), deepcopy(v, memo)) for k, v in self.items()])

  def __repr__(self):
    return json.dumps(self, indent=4, sort_keys=True)


def recursive_objectify(nested_dict):
  "Turns a nested_dict into a nested ParamDict"
  result = deepcopy(nested_dict)
  for k, v in result.items():
    if isinstance(v, Mapping):
      result[k] = recursive_objectify(v)
  return ParamDict(result)


class SafeDict(dict):
  """ A dict with prohibiting init from a list of pairs containing duplicates"""
  def __init__(self, *args, **kwargs):
    if args and args[0] and not isinstance(args[0], dict):
      keys, _ = zip(*args[0])
      duplicates =[item for item, count in collections.Counter(keys).items() if count > 1]
      if duplicates:
        raise TypeError("Keys {} repeated in json parsing".format(duplicates))
    super().__init__(*args, **kwargs)

def load_json(file):
  """ Safe load of a json file (doubled entries raise exception)"""
  with open(file, 'r') as f:
    data = json.load(f, object_pairs_hook=SafeDict)
  return data


def update_recursive(d, u, defensive=False):
  for k, v in u.items():
    if defensive and k not in d:
      raise KeyError("Updating a non-existing key")
    if isinstance(v, Mapping):
      d[k] = update_recursive(d.get(k, {}), v)
    else:
      d[k] = v
  return d

def is_json_file(cmd_line):
  try:
    return os.path.isfile(cmd_line)
  except Exception as e:
    warn('JSON parsing suppressed exception: ', e)
    return False


def is_parseable_dict(cmd_line):
  try:
    res = ast.literal_eval(cmd_line)
    return isinstance(res, dict)
  except Exception as e:
    warn('Dict literal eval suppressed exception: ', e)
    return False

def update_params_from_cmdline(cmd_line=None, default_params=None, custom_parser=None, verbose=True):
  """ Updates default settings based on command line input.

  :param cmd_line: Expecting (same format as) sys.argv
  :param default_params: Dictionary of default params
  :param custom_parser: callable that returns a dict of params on success
  and None on failure (suppress exceptions!)
  :param verbose: Boolean to determine if final settings are pretty printed
  :return: Immutable nested dict with (deep) dot access. Priority: default_params < default_json < cmd_line
  """
  if not cmd_line:
    cmd_line = sys.argv

  if default_params is None:
    default_params = {}

  if len(cmd_line) < 2:
    cmd_params = {}
  elif custom_parser and custom_parser(cmd_line):  # Custom parsing, typically for flags
    cmd_params = custom_parser(cmd_line)
  elif len(cmd_line) == 2 and is_json_file(cmd_line[1]):
    cmd_params = load_json(cmd_line[1])
  elif len(cmd_line) == 2 and is_parseable_dict(cmd_line[1]):
    cmd_params = ast.literal_eval(cmd_line[1])
  else:
    raise ValueError('Failed to parse command line')

  update_recursive(default_params, cmd_params)

  if JSON_FILE_KEY in default_params:
    json_params = load_json(default_params[JSON_FILE_KEY])
    if 'default_json' in json_params:
      json_base = load_json(json_params[JSON_FILE_KEY])
    else:
      json_base = {}
    update_recursive(json_base, json_params)
    update_recursive(default_params, json_base)

  update_recursive(default_params, cmd_params)
  final_params = recursive_objectify(default_params)
  if verbose:
    print(final_params)

  update_params_from_cmdline.start_time = time.time()
  return final_params

update_params_from_cmdline.start_time = None