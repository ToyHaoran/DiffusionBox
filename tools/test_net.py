# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from mega_core.utils.env import setup_environment  # noqa F401 isort:skip
from mega_core.utils.dist_env import init_dist

import argparse
import os

import torch
from mega_core.config import cfg
from mega_core.data import make_data_loader
from mega_core.engine.inference import inference
from mega_core.modeling.detector import build_detection_model
from mega_core.utils.checkpoint import DetectronCheckpointer
from mega_core.utils.collect_env import collect_env_info
from mega_core.utils.comm import synchronize, get_rank
from mega_core.utils.logger import setup_logger
from mega_core.utils.miscellaneous import mkdir
from mega_core.modeling.detector.diffusion_det import add_diffusiondet_config

# Check if we can enable mixed-precision via apex.amp
# try:
#     from apex import amp
# except ImportError:
#     raise ImportError('Use APEX for mixed precision via apex.amp')


def main():
    #os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # for debugging
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        '--launcher',
        choices=['pytorch', 'mpi'],
        default='pytorch',
        help='job launcher')
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
    )
    parser.add_argument(
        "--motion-specific",
        "-ms",
        action="store_true",
        help="if True, evaluate motion-specific iou"
    )
    parser.add_argument("--master_port", "-mp", type=str, default='29999')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    if args.launcher == "pytorch":
        num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    elif args.launcher == "mpi":
        num_gpus = int(os.environ["OMPI_COMM_WORLD_SIZE"]) if "OMPI_COMM_WORLD_SIZE" in os.environ else 1
    else:
        num_gpus = 1
    distributed = num_gpus > 1

    if distributed:
        init_dist(args.launcher, args=args)
        synchronize()

    BASE_CONFIG = "configs/BASE_RCNN_{}gpu.yaml".format(num_gpus)
    cfg.merge_from_file(BASE_CONFIG)
    # if 'Diffusion' in args.config_file:
    add_diffusiondet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("mega_core", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    # Initialize mixed-precision if necessary
    use_mixed_precision = cfg.DTYPE == 'float16'
    # amp_handle = amp.init(enabled=use_mixed_precision, verbose=cfg.AMP_VERBOSE)

    output_dir = cfg.OUTPUT_DIR

    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None, flownet=None)

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            cfg,
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            motion_specific=args.motion_specific,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            bbox_aug=cfg.TEST.BBOX_AUG.ENABLED,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()


if __name__ == "__main__":
    main()
