# run.py  ─── add ability to load & test checkpoints
import os, yaml, argparse, torch, multiprocessing
from pathlib import Path
from lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

from models import *                 # vae_models dict
from experiment import VAEXperiment
from dataset import VAEDataset

# ─────────────────────────────────────────────────────────

def build_trainer(tb_logger, cfg, callbacks_extra=None) -> Trainer:
    """helper to build a Trainer with common callbacks"""
    cbs = [
        LearningRateMonitor(),
        ModelCheckpoint(
            save_top_k=2,
            dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
            monitor="val_loss",
            save_last=True),
        EarlyStopping(monitor="val_loss", patience=10, mode="min"),
    ]
    if callbacks_extra:
        cbs.extend(callbacks_extra)

    return Trainer(logger=tb_logger, callbacks=cbs, **cfg["trainer_params"])


def main() -> None:
    # ── argparse ──────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Run / eval a VAE")
    parser.add_argument("-c", "--config", default="configs/vae.yaml",
                        help="yaml config file")
    parser.add_argument("--ckpt", default=None,
                        help="path to .ckpt to evaluate (skip training)")
    args = parser.parse_args()

    # ── config & reproducibility ─────────────────────────
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    seed_everything(cfg["exp_params"]["manual_seed"], workers=True)

    # ── datamodule ───────────────────────────────────────
    dm = VAEDataset(**cfg["data_params"],
                    pin_memory=cfg["trainer_params"].get("accelerator", "") == "gpu")
    dm.setup()

    # ── logger ───────────────────────────────────────────
    tb_logger = TensorBoardLogger(
        save_dir=cfg["logging_params"]["save_dir"],
        name=cfg["model_params"]["name"])

    # ── build model or load from ckpt ────────────────────
    if args.ckpt is None:                      # ─── training run ──
        core = vae_models[cfg["model_params"]["name"]](**cfg["model_params"])
        experiment = VAEXperiment(core, cfg["exp_params"])
        trainer = build_trainer(tb_logger, cfg)
        Path(f"{tb_logger.log_dir}/Samples").mkdir(parents=True, exist_ok=True)
        Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(parents=True, exist_ok=True)
        print(f"======= Training {cfg['model_params']['name']} =======")
        trainer.fit(experiment, datamodule=dm)

    else:                                      # ─── evaluation run ──
        print(f"Loading checkpoint: {args.ckpt}")
        experiment = VAEXperiment.load_from_checkpoint(
            args.ckpt,
            vae_model=vae_models[cfg["model_params"]["name"]](**cfg["model_params"]),
            params=cfg["exp_params"],
        )

        # need a trainer so hooks (on_validation_end) have .trainer attribute
        trainer = build_trainer(tb_logger, cfg, callbacks_extra=[])
        experiment.trainer = trainer  # manually attach

        # trainer.datamodule = dm
        # experiment.visualize_latent_space(experiment.trainer.datamodule.train_dataloader(), method="tsne", title="Train Latent Space")
        # experiment.visualize_latent_space(experiment.trainer.datamodule.test_dataloader(), method="tsne", title="Test Latent Space")

        # Option 1: run one validation epoch → triggers sample_images_next_to_origs
        #trainer.validate(experiment, datamodule=dm, verbose=False)

        # Option 2 (manual): just call helper once
        #experiment.sample_images_next_to_origs()

        val_loader = dm.val_dataloader()
        test_loader = dm.test_dataloader()
        experiment.classify_cracked_images([val_loader, test_loader],
                                           result_dir="crack_results",
                                           k=3.0)

# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    multiprocessing.freeze_support()   # for Windows dataloader workers
    main()
    #todo: python load_and_run_model.py -c configs/vae.yaml --ckpt .\logs\VanillaVAE\version_13\checkpoints\epoch=96-step=24250.ckpt
