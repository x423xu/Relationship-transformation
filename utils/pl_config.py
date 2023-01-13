
def set_arguments_pl(args):
    pl_args = {}
    accelerator_cfg = get_accelerator_cfg(args)
    train_val_cfg = get_train_val_cfg(args)
    test_cfg = get_test_cfg(args)
    logger_cfg = get_logger_cfg(args)
    checkpoint_cfg = get_checkpoint_cfg(args)
    resume_cfg = get_resume_cfg(args)

    pl_args.update(**accelerator_cfg, **train_val_cfg, **test_cfg, **logger_cfg, **checkpoint_cfg, **resume_cfg)
    return pl_args

'''
configurations for accelerator: single gpu or ddp
'''
def get_accelerator_cfg(args):
    return None

'''
configurations for train validation
'''
def get_train_val_cfg(args):
    return None

'''
configurations for test
'''
def get_test_cfg(args):
    return None

'''
configurations for logger
'''

def get_logger_cfg(args):
    return None


'''
configurations for checkpoint
'''
def get_checkpoint_cfg(args):
    return None

'''
configurations for resume training
'''
def get_resume_cfg(args):
    return None