import os
import yaml
import argparse
import numpy as np
from pathlib import Path

from lightning import seed_everything

from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
import lightning as PL
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
#from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
#from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.strategies import DDPStrategy

from dataset import VAEDataset


def main():
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    devNumber = torch.cuda.current_device()
    print("Current device number is:", devNumber)
    devName = torch.cuda.get_device_name(devNumber)
    print("Current device name is:", devName)




    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help =  'path to the config file',
                        default='configs/vae.yaml')

    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)


    tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                                   name=config['model_params']['name'],)

    # For reproducibility
    seed_everything(config['exp_params']['manual_seed'], True)

    model = vae_models[config['model_params']['name']](**config['model_params'])
    experiment = VAEXperiment(model,
                              config['exp_params'])

    data = VAEDataset(**config["data_params"], pin_memory = config['trainer_params'].get('accelerator', '') == 'gpu')

    data.setup()
    runner = Trainer(logger=tb_logger,
                     callbacks=[
                         LearningRateMonitor(),
                         ModelCheckpoint(save_top_k=2,
                                         dirpath =os.path.join(tb_logger.log_dir , "checkpoints"),
                                         monitor= "val_loss",
                                         save_last= True),
                         EarlyStopping(
                             monitor="val_loss",
                             patience=10,  # stop if val_loss doesn't improve for 10 epochs
                             mode="min"  # minimize val_loss
                         )
                     ],
                     #strategy=DDPStrategy(find_unused_parameters=False),
                    # accelerator="gpu",
                    # devices=[0],
                     **config['trainer_params'])


    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment, datamodule=data)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # recommended on Windows
    main()