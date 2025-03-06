from capsnet.model import CapsNet
from capsnet.train_helper import train_capsnet
from capsnet.dataset_helper import CacheLoader
import logging
import sys
import pickle
import optuna
import os

def load_study(study_name):
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    os.makedirs(f"studies/{study_name}", exist_ok=True)

    storage_name = f"sqlite:///studies/{study_name}/hist.db"

    if os.path.exists(f"studies/{study_name}/sampler.pkl"):
        sampler = pickle.load(open(f"studies/{study_name}/sampler.pkl", "rb"))
    else:
        sampler = None
        
    if os.path.exists(f"studies/{study_name}/pruner.pkl"):
        pruner = pickle.load(open(f"studies/{study_name}/pruner.pkl", "rb"))
    else:
        pruner = None

    return optuna.create_study(direction="maximize", study_name=study_name, storage=storage_name, sampler=sampler, pruner=pruner, load_if_exists=True)

def save_study(study_name, study):
    with open(f"studies/{study_name}/sampler.pkl", "wb") as f:
        pickle.dump(study.sampler, f)

    with open(f"studies/{study_name}/pruner.pkl", "wb") as f:
        pickle.dump(study.pruner, f)

def objective(trial, epochs, device, cache_loader: CacheLoader):
    input_size = trial.suggest_categorical("input_size", [32, 64, 128])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    alpha = trial.suggest_float("alpha", 1e-4, 1e-2, log=True)
    routing_iters = trial.suggest_int("routing_iters", 1, 5)
    num_capsules = trial.suggest_int("num_capsules", 16, 64)
    capsule_dim = trial.suggest_int("capsule_dim", 4, 16)
    primary_kernel_size = trial.suggest_int("primary_kernel_size", 3, 9)
    primary_stride = trial.suggest_int("primary_stride", 1, 3)
    
    n_decoder_layers = trial.suggest_int("n_decoder_layers", 1, 2)
    decoder_hidden_dims = []
    decoder_dropout_rates = []
    for i in range(n_decoder_layers):
        hidden_dim = trial.suggest_categorical(f"decoder_hidden_dim_{i}", [128, 256, 512, 1024, 2048, 4096])
        dropout_rate = trial.suggest_float(f"decoder_dropout_rate_{i}", 0.2, 0.5)
        decoder_hidden_dims.append(hidden_dim)
        decoder_dropout_rates.append(dropout_rate)
    final_dropout = trial.suggest_float("decoder_final_dropout", 0.2, 0.5)
    decoder_dropout_rates.append(final_dropout)
    
    primary_caps_params = {
        'num_capsules': num_capsules,
        'capsule_dim': capsule_dim,
        'kernel_size': primary_kernel_size,
        'stride': primary_stride
    }
    
    train_loader, val_loader = cache_loader.get_or_create_cache(img_size=input_size)
    
    try: # model might not compile with random hyperparameters
        model = CapsNet(
            input_size=input_size,
            conv_channels=[64, 128, 256],
            primary_caps_params=primary_caps_params,
            num_classes=6,
            capsule_out_dim=16,
            routing_iters=routing_iters,
            decoder_hidden_dims=decoder_hidden_dims,
            decoder_dropout_rates=decoder_dropout_rates
        ).to(device)
    except:
        return 0
    
    model, f1_top3 = train_capsnet(model, train_loader, val_loader, device, epochs=epochs, alpha=alpha, lr=lr)

    return f1_top3

def study_to_model_params(study_params, conv_channels, num_classes, capsule_out_dim):
    best_primary_caps_params = {
        'num_capsules': study_params["num_capsules"],
        'capsule_dim': study_params["capsule_dim"],
        'kernel_size': study_params["primary_kernel_size"],
        'stride': study_params["primary_stride"]
    }
    
    return {
        "input_size": study_params["input_size"],
        "conv_channels": conv_channels,
        "primary_caps_params": best_primary_caps_params,
        "num_classes": num_classes,
        "capsule_out_dim": capsule_out_dim,
        "routing_iters": study_params["routing_iters"],
        "decoder_hidden_dims": [v for param, v in study_params.items() if "decoder_hidden_dim_" in param],
        "decoder_dropout_rates": [v for param, v in study_params.items() if "decoder" in param and "dropout" in param]
    }, study_params["alpha"], study_params["lr"]