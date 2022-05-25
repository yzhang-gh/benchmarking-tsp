from others import DotDict


def get_pomo_solver_options(graph_size, model_path, epoch, num_data_aug=1):
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
    opts = DotDict(
        {
            ## Overall settings
            "problem": "tsp",
            "graph_size": graph_size,
            "dummy_rate": 0.5,
            "step_method": "2_opt",
            "init_val_met": "random",
            "no_cuda": False,
            "no_tb": True,
            "show_figs": False,
            "no_saving": True,
            "use_assert": False,
            "no_DDP": True,
            ## DACT parameters
            "v_range": 6.0,
            "DACTencoder_head_num": 4,
            "DACTdecoder_head_num": 4,
            "critic_head_num": 6,
            "embedding_dim": 64,
            "hidden_dim": 64,
            "n_encode_layers": 3,
            "normalization": "layer",
            ## Training parameters
            "RL_agent": "ppo",
            "gamma": 0.999,
            "K_epochs": 3,
            "eps_clip": 0.1,
            "T_train": 200,
            "n_step": 4,
            "best_cl": False,
            "Xi_CL": 0.25,
            "batch_size": 600,
            "epoch_end": 200,
            "epoch_size": 12000,
            "lr_model": 0.0001,
            "lr_critic": 3e-05,
            "lr_decay": 0.985,
            "max_grad_norm": 0.04,
            ## Inference and validation parameters
            "T_max": T_max,
            "eval_only": True,
            # "val_size": 10000,
            # "val_dataset": "./datasets/tsp_100_10000.pkl",
            "val_m": num_data_aug,
            ## Resume and load models
            "load_path": model_path,
            "resume": None,
            "epoch_start": 0,
            ## Logs/output settings
            "no_progress_bar": False,
            "log_dir": "logs",
            "log_step": 50,
            "output_dir": "outputs",
            "run_name": "run_name",
            "checkpoint_epochs": 1,
            "world_size": 1,
            "distributed": False,
            "use_cuda": True,
            "P": 250,
            "save_dir": "",
        }
    )
    opts.device = "cuda"
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
