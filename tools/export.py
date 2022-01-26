from __future__ import absolute_import, division, print_function

import argparse
import torch

import _init_paths
from config import cfg, update_config
from detectors.detector_factory import detector_factory


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--TESTMODEL',
                        help='model directory',
                        type=str,
                        default='')
    args = parser.parse_args()

    return args


def demo(cfg):
    Detector = detector_factory[cfg.TEST.TASK]
    detector = Detector(cfg)
    m = detector.model
    shape = (1, 3, cfg.MODEL.INPUT_RES, cfg.MODEL.INPUT_RES)
    torch.onnx.export(m, torch.randn(shape).cuda(),
                      f'{cfg.EXP_ID}.onnx', opset_version=11)
    print("Done!")


if __name__ == '__main__':
    args = parse_args()
    update_config(cfg, args.cfg)
    cfg.defrost()
    cfg.TEST.MODEL_PATH = args.TESTMODEL
    cfg.freeze()
    demo(cfg)
