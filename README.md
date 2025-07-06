### Installation
```
$ pip install -r requirements.txt 

(torch has to be installed manually (with CUDA support if GPU is available)
```

### Usage
```
$ cd PyTorch-VAE
$ python load_and_run_model.py -c configs/vae.yaml --ckpt .\logs\VanillaVAE\<insert version to load>\checkpoints\<insert checkpoint file to load>
