# Eb-LAIfO
Event-based Latent Adversarial Imitation from Observations

## Instructions

### Use anaconda to create a virtual environment

**Step 1.** Install miniconda

```shell
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

**Step 2.** Install [MuJoCo](https://github.com/deepmind/mujoco)

**Step 3.** Clone repo and create conda environment

```shell
conda env create -f environment.yml
conda activate VRL3
```

### Expert policies

Download the policies [here](https://figshare.com/s/c441615a51a79a22c3e4) and unzip in main directory.

### Train RL+imitation from expert videos with visual mismatch

#### Door-Color

```shell
python train_RL_with_expert.py seed=0 task_agent=door_color task_expert=door pretrained_encoder=false save_video=true RL_plus_IL=true GAN_loss=bce apply_aug='CL-Q' aug_type='color' CL_data_type=agent save_models=true
```

#### Door-Light

```shell
python train_RL_with_expert.py seed=0 task_agent=door_light task_expert=door pretrained_encoder=false save_video=true RL_plus_IL=true GAN_loss=bce apply_aug='CL-Q' aug_type='brightness' CL_data_type=agent save_models=true
```

#### Hammer-Color

```shell
python train_RL_with_expert.py seed=0 task_agent=hammer_color task_expert=hammer pretrained_encoder=false save_video=true RL_plus_IL=true GAN_loss=bce apply_aug='CL-Q' aug_type='color' CL_data_type=agent save_models=true 
```

#### Hammer-Light

```shell
python train_RL_with_expert.py seed=0 task_agent=hammer_light task_expert=hammer pretrained_encoder=false save_video=true RL_plus_IL=true GAN_loss=bce apply_aug='CL-Q' aug_type='brightness' CL_data_type=agent save_models=true
```

#### Pen-Color

```shell
python train_RL_with_expert.py seed=0 task_agent=pen_color task_expert=pen pretrained_encoder=false save_video=true RL_plus_IL=true GAN_loss=bce apply_aug='CL-Q' aug_type='color' CL_data_type=agent save_models=true 
```

#### Pen-Light

```shell
python train_RL_with_expert.py seed=0 task_agent=pen_light task_expert=pen pretrained_encoder=false save_video=true RL_plus_IL=true GAN_loss=bce apply_aug='CL-Q' aug_type='brightness' CL_data_type=agent save_models=true
```
