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
python train_LAIL_MI.py task_agent=walker_walk task_expert=walker_walk agent=lail_cl_multiset difficulty=easy delta_source=0.2 delta_target=-0.25 aug_type='brightness'
```

**Eb-LAIfO**

```shell
python train_LAIL_MI.py task_agent=walker_walk task_expert=walker_walk agent=lail_eb_multiset difficulty=easy delta_source=0.2 delta_target=-0.25 frame_stack=4
```

#### Body

**C-LAIfO**

```shell
python train_LAIL_MI.py task_agent=walker_walk task_expert=walker_walk agent=lail_cl_multiset difficulty=color_body aug_type='color'
```

**Eb-LAIfO**

```shell
python train_LAIL_MI.py task_agent=walker_walk task_expert=walker_walk agent=lail_eb_multiset difficulty=color_body frame_stack=4
```

#### Floor

**C-LAIfO**

```shell
python train_LAIL_MI.py task_agent=walker_walk task_expert=walker_walk agent=lail_cl_multiset difficulty=color_floor aug_type='color'
```

**Eb-LAIfO**

```shell
python train_LAIL_MI.py task_agent=walker_walk task_expert=walker_walk agent=lail_eb_multiset difficulty=color_body frame_stack=4
```

#### Background

**C-LAIfO**

```shell
python train_LAIL_MI.py task_agent=walker_walk task_expert=walker_walk agent=lail_cl_multiset difficulty=color_bg aug_type='color'
```

**Eb-LAIfO**

```shell
python train_LAIL_MI.py task_agent=walker_walk task_expert=walker_walk agent=lail_eb_multiset difficulty=color_body frame_stack=4
```

#### Full

**C-LAIfO**

```shell
python train_LAIL_MI.py task_agent=walker_walk task_expert=walker_walk agent=lail_cl_multiset difficulty=color_all_together aug_type='color'
```

**Eb-LAIfO**

```shell
python train_LAIL_MI.py task_agent=walker_walk task_expert=walker_walk agent=lail_eb_multiset difficulty=color_body frame_stack=4
```