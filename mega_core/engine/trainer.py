# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import os
import time

import torch
import torch.distributed as dist
from tqdm import tqdm

from mega_core.data import make_data_loader
from mega_core.utils.comm import get_world_size, synchronize
from mega_core.utils.metric_logger import MetricLogger
from mega_core.engine.inference import inference

# from apex import amp
from torch.cuda.amp import autocast, GradScaler

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    cfg,
    model,
    data_loader,
    data_loader_val,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    test_period,
    arguments,
    tensorboard_writer
):
    logger = logging.getLogger("mega_core.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"] * cfg.SOLVER.ACCUMULATION_STEPS
    model.train()
    start_training_time = time.time()
    end = time.time()
    # 设置IoU类型和数据集名称
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    dataset_names = cfg.DATASETS.TEST

    # 初始化混合精度训练的 GradScaler 对象
    scaler = GradScaler()

    optimizer.zero_grad()
    # 训练循环
    for iter, (images, targets, _) in enumerate(data_loader, start_iter):
        # 数据处理
        if any(len(target) < 1 for target in targets):
            logger.error(f"Iteration={iter + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}" )
            continue
        data_time = time.time() - end
        iter = iter + 1  # iter for lataloader
        iteration = iter // cfg.SOLVER.ACCUMULATION_STEPS + 1  # this is real iteration
        arguments["iteration"] = iteration - 1

        if not cfg.MODEL.VID.ENABLE:
            images = images.to(device)
        else:  # 选择使用视频目标检测相关的处理逻辑
            method = cfg.MODEL.VID.METHOD
            if method in ("base", ):
                images = images.to(device)
            elif method in ("rdn", "mega", "dafa", "diffusion", "fgfa", "dff"):
                images["cur"] = images["cur"].to(device)
                for key in ("ref", "ref_l", "ref_m", "ref_g"):
                    if key in images.keys():
                        images[key] = [img.to(device) for img in images[key]]
            else:
                raise ValueError("method {} not supported yet.".format(method))
        if method in ("mega", "dafa", "diffusion"):  # 进一步处理目标，将其全部转到cuda上
            targets_c, targets_g, targets_l = targets[0]
            targets_c = [target.to(device) for target in targets_c]  # 当前帧
            targets_g = [tg.to(device) for tg in targets_g]  # 全局参考帧
            targets_l = [tl.to(device) for tl in targets_l]  # 局部参考帧
            targets = [targets_c, targets_g, targets_l]
        else:
            targets = [target.to(device) for target in targets]

        # 确定需要在一个批次内重复使用当前帧，和具有至少一个目标的全局参考帧的次数。
        if method in ("mega", "dafa", "diffusion"):
            # 对于每个全局参考帧的目标信息，统计其边界框的数量，并将结果存储在 num_boxes_targets 列表中。
            num_boxes_targets = [len(target.bbox) for target in targets_g]
            # 确定哪些全局参考帧需要被重复使用
            idxs = [-1] + [i for i, x in enumerate(num_boxes_targets) if x > 0]
            # 确定总共需要多少次重复使用
            total_reuse_count = min(cfg.SOLVER.BATCH_REUSE_STEPS, len(idxs))
            if len(targets_g) <= 1:
                total_reuse_count = 1
        else:
            idxs = [-1]
            total_reuse_count = 1
        for i in range(total_reuse_count):
            idx = idxs[i]
            if idx != -1 and method in ("mega", "dafa", "diffusion"):
                # 1. 随机选择全局参考帧
                images["cur"], images["ref_g"][idx].tensors = images["ref_g"][idx].tensors[0], images["cur"][None,:]
                targets[0][0], targets[1][idx] = targets_g[idx], targets_c[0]
            # 使用交换后的图像和目标信息进行模型的前向传播，得到损失。
            loss_dict = model(images, targets)
            # 对损失进行处理，使其除以总的重复使用次数，以便进行累积梯度的平均。
            losses = sum(loss for loss in loss_dict.values()) / (cfg.SOLVER.ACCUMULATION_STEPS * total_reuse_count)

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)  # 对损失进行降维，以便进行日志记录和展示。
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss=losses_reduced, **loss_dict_reduced)  # 使用 MetricLogger 对损失进行更新

            # Note: If mixed precision is not used, this ends up doing nothing
            # Otherwise apply loss scaling for mixed-precision recipe
            # with amp.scale_loss(losses, optimizer) as scaled_losses:
            #     scaled_losses.backward()
            # 这是因为半精度的数值范围有限，因此需要用它放大
            scaler.scale(losses).backward()

            # unscale之前放大后的梯度，但是scale太多可能出现inf或NaN, 故其会判断是否出现了inf/NaN
            # 如果梯度的值不是 infs 或者 NaNs, 那么调用optimizer.step()来更新权重,
            # 如果检测到出现了inf或者NaN，就跳过这次梯度更新，同时动态调整scaler的大小
            scaler.step(optimizer)

            # 查看是否要更新scaler,这个要注意不能丢
            scaler.update()

        # 在每次累积梯度达到指定步数时更新模型权重，并进行一些日志记录和展示。
        if iter % cfg.SOLVER.ACCUMULATION_STEPS == 0:
            optimizer.step()  # 用于根据当前累积的梯度更新模型的权重
            optimizer.zero_grad()
            # 学习率调度器更新
            if cfg.SOLVER.LR_SCHEDULER_TYPE == "cosine":
                scheduler.step_update(iter // cfg.SOLVER.ACCUMULATION_STEPS)
            else:
                scheduler.step()
        # 统计时间相关信息
        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)
        # 计算并记录预估的剩余训练时间
        eta_seconds = meters.time.global_avg * (max_iter - iter)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        # 迭代20次就定期输出日志和在 TensorBoard中记录
        if (iter % (20 * cfg.SOLVER.ACCUMULATION_STEPS) == 0 or iter == max_iter) and torch.cuda.current_device() == 0:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
            if tensorboard_writer is not None:
                for key, val in meters.meters.items():
                    if 'loss' in key.lower():
                        tensorboard_writer.add_scalar('Train/' + key,
                                                      val.global_avg, iteration)
                        tensorboard_writer.add_scalar('Train_Avg20/' + key,
                                                      val.avg, iteration)
                tensorboard_writer.add_scalar('Train/RunningLearningRate',
                                              optimizer.param_groups[-1]['lr'], iteration)
                                              #scheduler.get_last_lr()[0], iteration)
        if iter % checkpoint_period == 0:  # 模型检查点保存
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iter == max_iter:
            checkpointer.save("model_final", **arguments)
        # 验证集性能评估
        if data_loader_val is not None and test_period > 0 and (iter % test_period == 0 or iter == max_iter):
            meters_val = MetricLogger(delimiter="  ")
            synchronize()
            val_result = inference(  # The result can be used for additional logging, e. g. for TensorBoard
                cfg,
                model,
                # The method changes the segmentation mask format in a data loader,
                # so every time a new data loader is created:
                # make_data_loader(cfg, is_train=False, is_distributed=(get_world_size() > 1), is_for_period=False)[0],
                data_loader_val[0],
                dataset_name="[Validation]",
                iou_types=iou_types,
                box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                device=cfg.MODEL.DEVICE,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                output_folder=None,
            )
            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar('Val/mAP', val_result[0]['map'], iteration)
            synchronize()
            model.train()
            '''
            with torch.no_grad():
                # Should be one image for each GPU:
                for iteration_val, (images_val, targets_val, _) in enumerate(tqdm(data_loader_val)):
                    images_val = images_val.to(device)
                    targets_val = [target.to(device) for target in targets_val]
                    loss_dict = model(images_val, targets_val)
                    losses = sum(loss for loss in loss_dict.values())
                    loss_dict_reduced = reduce_loss_dict(loss_dict)
                    losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                    meters_val.update(loss=losses_reduced, **loss_dict_reduced)
            synchronize()
            logger.info(
                meters_val.delimiter.join(
                    [
                        "[Validation]: ",
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters_val),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
            '''
        if iter == max_iter:
            break

    # 训练结束，统计时间
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
