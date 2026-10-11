# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# ------------------------------------------------------------------------------


import torch
import torch.nn as nn

from .dist_helper import compact_logging_enabled, log


def get_optimizer(config, model, lr_scaler):
    params = []
    compact_log = compact_logging_enabled()
    trainable_tensors = 0
    trainable_parameters = 0
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        trainable_tensors += 1
        trainable_parameters += value.numel()
        lr = config.LR.LR * lr_scaler
        if '_nwd' in key or key.endswith('.bias') or len(value.shape) == 1:
            weight_decay = 0.
        else:
            weight_decay = config.OPTIMIZER.WD

        if not compact_log:
            log(f'Params: {key}, LR: {lr}, Weight_Decay: {weight_decay}')
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if compact_log:
        log(
            f'Optimizer: type={config.OPTIMIZER.TYPE}, '
            f'lr={config.LR.LR * lr_scaler:.8g}, '
            f'trainable_tensors={trainable_tensors}, '
            f'trainable_params={trainable_parameters:,}'
        )

    if config.OPTIMIZER.TYPE == 'sgd':
        optimizer = torch.optim.SGD(params,
                                    lr=config.LR.LR * lr_scaler,
                                    momentum=config.OPTIMIZER.MOMENTUM,
                                    weight_decay=config.OPTIMIZER.WD,
                                    )
    elif config.OPTIMIZER.TYPE == 'adam':
        optimizer = torch.optim.Adam(params,
                                     lr=config.LR.LR * lr_scaler,
                                     )
    elif config.OPTIMIZER.TYPE == 'adamw':
        optimizer = torch.optim.AdamW(params,
                                      lr=config.LR.LR * lr_scaler,
                                      weight_decay=config.OPTIMIZER.WD,
                                      )
    else:
        raise NotImplementedError

    return optimizer
