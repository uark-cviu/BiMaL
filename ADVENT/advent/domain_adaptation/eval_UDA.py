# --------------------------------------------------------
# Domain adpatation evaluation
# Copyright (c) 2019 valeo.ai
#
# Written by Tuan-Hung Vu
# --------------------------------------------------------

import os.path as osp
import time
import os

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from advent.utils.func import per_class_iu, fast_hist
from advent.utils.serialization import pickle_dump, pickle_load
from advent.utils.viz_segmask import colorize_mask
from advent.utils.func import prob_2_entropy
from PIL import Image
import torch.nn.functional as F

def evaluate_domain_adaptation( models, test_loader, cfg,
                                fixed_test_size=True,
                                verbose=True):
    device = cfg.GPU_ID
    interp = None
    if fixed_test_size:
        interp = nn.Upsample(size=(cfg.TEST.OUTPUT_SIZE_TARGET[1], cfg.TEST.OUTPUT_SIZE_TARGET[0]), mode='bilinear', align_corners=True)
    # eval
    if cfg.TEST.MODE == 'single':
        eval_single(cfg, models,
                    device, test_loader, interp, fixed_test_size,
                    verbose)
    elif cfg.TEST.MODE == 'best':
        eval_best(cfg, models,
                  device, test_loader, interp, fixed_test_size,
                  verbose)
    else:
        raise NotImplementedError(f"Not yet supported test mode {cfg.TEST.MODE}")


def eval_single(cfg, models,
                device, test_loader, interp,
                fixed_test_size, verbose):
    assert len(cfg.TEST.RESTORE_FROM) == len(models), 'Number of models are not matched'
    for checkpoint, model in zip(cfg.TEST.RESTORE_FROM, models):
        load_checkpoint_for_evaluation(model, checkpoint, device)
    # eval
    hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
    for index, batch in tqdm(enumerate(test_loader)):
        image, label, _, name = batch
        if not fixed_test_size:
            interp = nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True)
        with torch.no_grad():
            output = None
            for model, model_weight in zip(models, cfg.TEST.MODEL_WEIGHT):
                pred_main = model(image.cuda(device))[1]
                output_ = interp(pred_main).cpu().data[0].numpy()
                if output is None:
                    output = model_weight * output_
                else:
                    output += model_weight * output_
            assert output is not None, 'Output is None'
            output = output.transpose(1, 2, 0)
            output = np.argmax(output, axis=2)
        label = label.numpy()[0]
        hist += fast_hist(label.flatten(), output.flatten(), cfg.NUM_CLASSES)
    inters_over_union_classes = per_class_iu(hist)
    print(f'mIoU = \t{round(np.nanmean(inters_over_union_classes) * 100, 2)}')
    if verbose:
        display_stats(cfg, test_loader.dataset.class_names, inters_over_union_classes)

def concat_img(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, min(im1.height, im2.height)))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def save_image(image_path, in_img, gt_seg, pr_seg, en_img):
    in_img = Image.fromarray(in_img, 'RGB')
    #en_img = en_img.convert('RGB')
    pr_seg = pr_seg.convert('RGB')
    gt_seg = gt_seg.convert('RGB')


    image = concat_img(in_img, pr_seg)
    image = concat_img(image,  en_img)
    image = concat_img(image,  gt_seg)

    image.save(image_path)


def eval_best(cfg, models,
              device, test_loader, interp,
              fixed_test_size, verbose):
    assert len(models) == 1, 'Not yet supported multi models in this mode'
    assert osp.exists(cfg.TEST.SNAPSHOT_DIR[0]), 'SNAPSHOT_DIR is not found'
    start_iter = cfg.TEST.SNAPSHOT_STEP
    step = cfg.TEST.SNAPSHOT_STEP
    max_iter = cfg.TEST.SNAPSHOT_MAXITER
    cache_path = osp.join(cfg.TEST.SNAPSHOT_DIR[0], 'all_res.pkl')
    if osp.exists(cache_path):
        all_res = pickle_load(cache_path)
    else:
        all_res = {}
    cur_best_miou = -1
    cur_best_13_class_miou = -1
    cur_best_model = ''
    for i_iter in range(start_iter, max_iter + 1, step):
        restore_from = osp.join(cfg.TEST.SNAPSHOT_DIR[0],  f'model_{i_iter}.pth')
        if not osp.exists(restore_from):
            # continue
            if cfg.TEST.WAIT_MODEL:
                print('Waiting for %s..!' % restore_from)
                while not osp.exists(restore_from):
                    time.sleep(5)
        print("Evaluating model", restore_from)
        if cfg.TEST.SAVE_IMAGE:
            image_dir = osp.join(cfg.TEST.SNAPSHOT_DIR[0], 'images', f'epoch_{i_iter}')
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)


        if i_iter not in all_res.keys():
            load_checkpoint_for_evaluation(models[0], restore_from, device)
            # eval
            hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
            # for index, batch in enumerate(test_loader):
            #     image, _, _, name = batch
            test_iter = iter(test_loader)
            for index in tqdm(range(len(test_loader))):
                image, label, _, name = next(test_iter)
                if cfg.TEST.SAVE_IMAGE:
                    image_path = os.path.join(image_dir, "%05d.jpg" % index)
                if not fixed_test_size:
                    interp = nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True)
                with torch.no_grad():
                    pred_main = models[0](image.cuda(device))[1]
                    if cfg.TEST.SAVE_IMAGE:
                        entropy = prob_2_entropy(F.softmax(interp(pred_main))).sum(dim=1)
                        entropy = (entropy - entropy.min()) / (entropy.max() - entropy.min())
                        entropy = entropy * 255

                    output = interp(pred_main).cpu().data[0].numpy()

                    if cfg.TEST.SAVE_IMAGE:
                        entropy = entropy.cpu().data[0].numpy().astype(np.uint8)
                        entropy = Image.fromarray(entropy, 'L')

                    output = output.transpose(1, 2, 0)
                    output = np.argmax(output, axis=2)

                    if cfg.TEST.SAVE_IMAGE:
                        pred_seg = colorize_mask(output)
                        input_img = image[0].detach().cpu().numpy().transpose(1, 2, 0) + cfg.TEST.IMG_MEAN
                        input_img = input_img.astype(np.uint8)

                label = label.numpy()[0]
                if cfg.TEST.SAVE_IMAGE:
                    gt_seg = colorize_mask(label)

                if cfg.TEST.SAVE_IMAGE:
                    save_image(image_path, input_img, gt_seg, pred_seg, entropy)

                hist += fast_hist(label.flatten(), output.flatten(), cfg.NUM_CLASSES)
                if verbose and index > 0 and index % 100 == 0:
                    print('{:d} / {:d}: {:0.2f}'.format(
                        index, len(test_loader), 100 * np.nanmean(per_class_iu(hist))))
            inters_over_union_classes = per_class_iu(hist)
            all_res[i_iter] = inters_over_union_classes
            pickle_dump(all_res, cache_path)
        else:
            inters_over_union_classes = all_res[i_iter]
        computed_miou = round(np.nanmean(inters_over_union_classes) * 100, 2)
        if len(inters_over_union_classes) == 16:
            cur_13_class_miou = 0.0
            for ii in range(16):
                if ii not in [3, 4, 5]:
                    cur_13_class_miou += inters_over_union_classes[ii]
            cur_13_class_miou /= 13.
        else:
            cur_13_class_miou = 0.0

        if cur_best_miou < computed_miou:
            cur_best_miou = computed_miou
            cur_best_model = restore_from
            cur_best_13_class_miou = cur_13_class_miou

        print('\tCurrent mIoU:', computed_miou)
        print('\tCurrent 13 classes mIoU:', cur_13_class_miou)
        print('\tCurrent best model:', cur_best_model)
        print('\tCurrent best mIoU:', cur_best_miou)
        print('\tCurrent best 13 classes mIoU:', cur_best_13_class_miou)
        if verbose:
            display_stats(cfg, test_loader.dataset.class_names, inters_over_union_classes)


def load_checkpoint_for_evaluation(model, checkpoint, device):
    saved_state_dict = torch.load(checkpoint)
    model.load_state_dict(saved_state_dict)
    model.eval()
    model.cuda(device)


def display_stats(cfg, name_classes, inters_over_union_classes):
    for ind_class in range(cfg.NUM_CLASSES):
        print(name_classes[ind_class]
              + '\t' + str(round(inters_over_union_classes[ind_class] * 100, 2)))
