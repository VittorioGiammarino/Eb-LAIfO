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
conda activate AIL_w_DA
```

### Train expert

```shell
python train_expert.py task=walker_walk seed=0 agent=ddpg frame_skip=1
```
Create a new directory `expert_policies`, move the trained expert policy in `expert_policies`.

Alternatively, download the policies [here](https://figshare.com/s/22de566de2229068fb75) and unzip in main directory.

### Train imitation from expert videos with visual mismatch for the DMC suite

#### Light

**C-LAIfO**

```shell
python train_LAIL_MI.py task_agent=walker_walk task_expert=walker_walk agent=lail_cl_multiset difficulty=easy delta_source=0.2 delta_target=-0.25 apply_aug='CL-Q' aug_type='brightness' CL_data_type=agent
```

#### Body

**C-LAIfO**

```shell
python train_LAIL_MI.py task_agent=walker_walk task_expert=walker_walk agent=lail_cl_multiset difficulty=color_body apply_aug='CL-Q' aug_type='color' CL_data_type=agent
```

#### Floor

**C-LAIfO**

```shell
python train_LAIL_MI.py task_agent=walker_walk task_expert=walker_walk agent=lail_cl_multiset difficulty=color_floor apply_aug='CL-Q' aug_type='color' CL_data_type=agent
```

#### Background

**C-LAIfO**

```shell
python train_LAIL_MI.py task_agent=walker_walk task_expert=walker_walk agent=lail_cl_multiset difficulty=color_bg apply_aug='CL-Q' aug_type='color' CL_data_type=agent
```

#### Full

**C-LAIfO**

```shell
python train_LAIL_MI.py task_agent=walker_walk task_expert=walker_walk agent=lail_cl_multiset difficulty=color_all_together apply_aug='CL-Q' aug_type='color' CL_data_type=agent
```
