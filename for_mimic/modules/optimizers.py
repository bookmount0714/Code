import torch


# def build_optimizer(args, model):
#     ve_params = list(map(id, model.visual_extractor.parameters()))
#     ed_params = filter(lambda x: id(x) not in ve_params, model.parameters())
#     optimizer = getattr(torch.optim, args.optim)(
#         [{'params': model.visual_extractor.parameters(), 'lr': args.lr_ve},
#          {'params': ed_params, 'lr': args.lr_ed}],
#         weight_decay=args.weight_decay,
#         amsgrad=args.amsgrad
#     )
#     return optimizer

def build_optimizer(args, model):
    # 冻结 visual_extractor 的参数
    for param in model.visual_extractor.parameters():
        param.requires_grad = False

    # 获取 visual_extractor 的参数 id
    ve_params = list(map(id, model.visual_extractor.parameters()))

    # 获取除 visual_extractor 之外的其他参数
    ed_params = filter(lambda x: id(x) not in ve_params, model.parameters())

    # 创建优化器，使用不同学习率
    optimizer = getattr(torch.optim, args.optim)(
        [{'params': model.visual_extractor.parameters(), 'lr': args.lr_ve},  # 这里 lr_ve 可以设置为 0
         {'params': ed_params, 'lr': args.lr_ed}],
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad
    )

    return optimizer

def build_lr_scheduler(args, optimizer):
    lr_scheduler = getattr(torch.optim.lr_scheduler, args.lr_scheduler)(optimizer, args.step_size, args.gamma)
    return lr_scheduler
