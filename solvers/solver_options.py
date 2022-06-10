from others import DotDict
from solvers.dact.options import get_options


def get_pomo_solver_options(graph_size, model_path, epoch, num_data_aug=1):
    """https://github.com/yd-kwon/POMO/blob/835c8c06248ade886856f7f5d207fca3f6c63575/NEW_py_ver/TSP/POMO/test_n100.py"""

    env_params = {
        "problem_size": graph_size,
        "pomo_size": graph_size,
    }

    model_params = {
        "embedding_dim": 128,
        "sqrt_embedding_dim": 128 ** (1 / 2),
        "encoder_layer_num": 6,
        "qkv_dim": 16,
        "head_num": 8,
        "logit_clipping": 10,
        "ff_hidden_dim": 512,
        "eval_type": "argmax",
    }

    tester_params = {
        "use_cuda": True,
        "cuda_device_num": 0,
        "model_load": {
            "path": model_path,  # directory path of pre-trained model and log files saved.
            "epoch": epoch,  # epoch version of pre-trained model to laod.
        },
        "test_episodes": 100 * 1000,
        "test_batch_size": 10000,
        "augmentation_enable": True if num_data_aug > 1 else False,
        "aug_factor": num_data_aug,
        "aug_batch_size": 1000,
    }

    if tester_params["augmentation_enable"]:
        tester_params["test_batch_size"] = tester_params["aug_batch_size"]

    return env_params, model_params, tester_params


def get_dact_solver_options(graph_size, model_path, num_data_aug=1, T_max=1500):
    opts = get_options("")

    opts.no_tb = True
    opts.no_saving = True
    opts.no_DDP = True
    opts.eval_only = True
    opts.P = 250

    opts.graph_size = graph_size
    opts.load_path = model_path
    opts.val_m = num_data_aug
    opts.T_max = T_max

    return opts


def get_nlkh_solver_options(model_path, max_trials):
    opts = DotDict(
        {
            "model_path": model_path,
            "device": "cuda",
            "max_trials": max_trials,
            "parallelism": 1,
            "batch_size": 1,
        }
    )
    return opts
