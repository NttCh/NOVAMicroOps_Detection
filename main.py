#!/usr/bin/env python
"""
Main script to run training or testing based on the config settings.
"""

import os
import sys
import torch
import optuna
import pandas as pd
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt

# ---------------- GPU and Environment Settings ----------------
torch.set_float32_matmul_precision('high')  # Use high precision for Tensor Cores.
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # Disable oneDNN custom ops if desired.
# --------------------------------------------------------------

from config import cfg, BASE_SAVE_DIR
from utils import set_seed, thai_time, load_obj
from model import build_classifier
from train import (
    train_stage,
    repeated_cross_validation,
    continue_training,
    print_trial_thai_callback
)
from inference import (
    evaluate_model,
    predict_test_folder,
    evaluate_test_roc,
    apply_compose
)
from data import PatchClassificationDataset
from optuna.exceptions import TrialPruned
from pytorch_lightning import seed_everything


def main():
    """
    Main entry point for running the script.
    Depending on cfg.run_mode, either trains or tests a model.
    """

    global BASE_SAVE_DIR
    # Set random seed
    set_seed(cfg.training.seed)
    seed_everything(cfg.training.seed, workers=True)

    # Create the base directory for logs if it doesn't exist
    date_folder = thai_time().strftime("%Y%m%d")    
    BASE_SAVE_DIR = os.path.join(cfg.general.save_dir, date_folder)
    os.makedirs(BASE_SAVE_DIR, exist_ok=True)

    # Directory to store best model
    best_model_folder = os.path.join(BASE_SAVE_DIR, "best_model")
    os.makedirs(best_model_folder, exist_ok=True)

    if cfg.run_mode.lower() == "test":
        print("[Main] TEST ONLY MODE")

        if not cfg.pretrained_ckpt:
            print("Please provide a valid pretrained checkpoint for testing.")
            sys.exit(1)

        num_classes = 2
        model = build_classifier(cfg, num_classes)

        print(f"Loading pretrained checkpoint from {cfg.pretrained_ckpt}")
        state_dict = torch.load(cfg.pretrained_ckpt, map_location=torch.device("cpu"))
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            if k.startswith("model."):
                new_key = k[len("model."):]
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict, strict=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        valid_transforms = A.Compose([
        load_obj(aug["class_name"])(**aug["params"]) for aug in cfg.augmentation.valid.augs
    ])


        test_folder = cfg.test.folder_path
        output_excel = os.path.join(BASE_SAVE_DIR, "test_predictions.xlsx")
        predict_test_folder(model, test_folder, valid_transforms, output_excel, print_results=True, model_path="None")

        if cfg.test_csv != "None":
            print("[Main] Evaluating ROC curve on test CSV")
            evaluate_test_roc(model, cfg.test_csv, cfg.test.folder_path, valid_transforms)

        print("[Main] TEST ONLY option complete.")

    else:
        print("[Main] TRAINING MODE")
        detection_csv = cfg.data.detection_csv

        # In your main() function, before the tuning block, add:
        trial_models = []  # container to hold models from each trial

        # If in tuning mode + optuna
        if cfg.tuning_mode and cfg.use_optuna:
            def objective(trial):
                trial_cfg = cfg.copy()
                trial_cfg.tuning_mode = True  # ignore pretrained_ckpt
                optuna_params = trial_cfg.get("optuna", {}).get("params", {})

                # Adjust hyperparameters from the trial
                for param_name, param_info in optuna_params.items():
                    if param_name in ["rotation_limit", "color_jitter_strength"]:
                        continue
                    ptype = param_info["type"]
                    if ptype == "loguniform":
                        trial_cfg.optimizer.params[param_name] = trial.suggest_float(
                            param_name, param_info["min"], param_info["max"], log=True
                        )
                    elif ptype == "categorical":
                        trial_cfg.data.batch_size = trial.suggest_categorical(
                            param_name, param_info["values"]
                        )
                    elif ptype == "float":
                        trial_cfg.optimizer.params[param_name] = trial.suggest_float(
                            param_name, param_info["min"], param_info["max"]
                        )
                    elif ptype == "int":
                        trial_cfg.optimizer.params[param_name] = trial.suggest_int(
                            param_name, param_info["min"], param_info["max"]
                        )

                # Handle augmentation-specific parameters
                r_min = optuna_params.get("rotation_limit", {}).get("min", 5)
                r_max = optuna_params.get("rotation_limit", {}).get("max", 15)
                c_min = optuna_params.get("color_jitter_strength", {}).get("min", 0.1)
                c_max = optuna_params.get("color_jitter_strength", {}).get("max", 0.3)

                rot = trial.suggest_int("rotation_limit", r_min, r_max)
                cj  = trial.suggest_float("color_jitter_strength", c_min, c_max)

                for aug in trial_cfg.augmentation.train.augs:
                    if aug["class_name"] == "albumentations.Rotate":
                        aug["params"]["limit"] = rot
                    elif aug["class_name"] == "albumentations.ColorJitter":
                        aug["params"]["brightness"] = cj
                        aug["params"]["contrast"]   = cj

                # Run training/CV
                trial_cfg.trainer.max_epochs = trial_cfg.training.tuning_epochs_detection
                num_classes = 2
                if trial_cfg.get("use_cv", False):
                    model_cv, score = repeated_cross_validation(
                        trial_cfg,
                        trial_cfg.data.detection_csv,
                        num_classes,
                        "detection",
                        repeats=trial_cfg.training.repeated_cv
                    )
                else:
                    model_cv, score = train_stage(
                        trial_cfg,
                        trial_cfg.data.detection_csv,
                        num_classes,
                        "detection"
                    )

                # Append the trained model for later selection
                trial_models.append(model_cv)
                return score

            # Run the study
            study = optuna.create_study(direction="maximize")
            study.optimize(
                objective,
                n_trials=cfg.get("optuna", {}).get("n_trials", 1),
                show_progress_bar=True,
                callbacks=[print_trial_thai_callback]
            )

            print("[Optuna] Best trial value:", study.best_trial.value)

            # Pick the best model returned by the best trial
            best_idx        = study.best_trial.number
            detection_model = trial_models[best_idx]
            detection_metric = study.best_trial.value

            # Save trial history and best params as before
            eval_folder = os.path.join(BASE_SAVE_DIR, "eval")
            os.makedirs(eval_folder, exist_ok=True)
            study.trials_dataframe().to_excel(
                os.path.join(eval_folder, "optuna_trials.xlsx"), index=False
            )
            pd.DataFrame([{
                **study.best_trial.params,
                "best_value": study.best_trial.value
            }]).to_excel(
                os.path.join(eval_folder, "optuna_best_params.xlsx"), index=False
            )
        else:
            # Non-tuning fallback: run once
            if cfg.get("use_cv", False):
                detection_model, detection_metric = repeated_cross_validation(
                    cfg,
                    detection_csv,
                    2,
                    "detection",
                    repeats=cfg.training.repeated_cv
                )
            else:
                detection_model, detection_metric = train_stage(
                    cfg,
                    detection_csv,
                    2,
                    "detection",
                    trial_number=None
                )

        # Save the checkpoint
        detection_checkpoint = os.path.join(best_model_folder, "best_detection.ckpt")
        torch.save(detection_model.state_dict(), detection_checkpoint)
        print(f"[Main] Saved detection checkpoint to {detection_checkpoint}")

        # Continue training with best checkpotint if needed
        detection_model = continue_training(detection_model, cfg, detection_csv, 2, "detection", best_model_folder=best_model_folder )

        # Evaluate final model
        from inference import evaluate_model
        evaluate_model(detection_model, detection_csv, cfg, stage="Detection")

        # Optionally predict on a test folder
        valid_transforms = A.Compose([
        load_obj(aug["class_name"])(**aug["params"]) for aug in cfg.augmentation.valid.augs
    ])
        if str(cfg.test.folder_path).lower() != "none":
            test_folder = cfg.test.folder_path
            output_excel = os.path.join(BASE_SAVE_DIR, "test_predictions.xlsx")
            print("\n[Main] Predicting on test folder:")
            predict_test_folder(detection_model, test_folder, valid_transforms, output_excel, print_results=True, model_path=cfg.pretrained_ckpt)

        print("[Main] Process finished.")


if __name__ == "__main__":
    main()
