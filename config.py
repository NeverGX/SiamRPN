import numpy as np
class Config:
    # dataset related
    exemplar_size = 127                    # exemplar size
    instance_size = 255                    # instance size
    context_amount = 0.5                   # context amount
    gpu_id = 0                             # gpu_id
    response_sz = 17                       # response size
    total_stride = 8                       # total stride of backbone

    # training related
    pairs_per_video = 20
    frame_range = 100                      # frame range of choosing the instance
    train_batch_size = 32                  # training batch size
    valid_batch_size = 8                   # validation batch size
    train_num_workers = 8                  # number of workers of train dataloader
    valid_num_workers = 8                  # number of workers of validation dataloader
    lr = 2e-2                              # start learning rate of SGD
    end_lr = 1e-5                          # end learning rate of SGD
    momentum = 0.9                         # momentum of SGD
    weight_decay = 0.0005                  # weight decay of optimizator
    step_size = 5                          # step size of LR_Schedular
    gamma = 0.5                            # decay rate of LR_Schedular
    epoch = 50                             # total epoch
    seed = 1234                            # seed to sample training videos
    log_dir = './models/logs'              # log dirs
    max_translate = 6                      # max translation of random shift
    max_stretch = 0.05
    anchor_base_size = 8
    anchor_scales = np.array([8, ])
    anchor_ratios = np.array([0.33, 0.5, 1, 2, 3])
    anchor_num = len(anchor_scales) * len(anchor_ratios)
    clip = 10                              # grad clip
    lamb = 6
    pos_threshold = 0.6
    neg_threshold = 0.3
    num_pos = 16
    num_neg = 48
    sample_type = 'uniform'
    gray_ratio = 0.25
    color_ratio = 0.1
    blur_ratio = 0.15

    # tracking related

    scale_lr = 0.295  # scale learning rate 0.295
    window_influence = 0.42  # window influence default :0.42
    penalty_k = 0.055  # scale penalty 0.055
    z_lr = 0.012





config = Config()
