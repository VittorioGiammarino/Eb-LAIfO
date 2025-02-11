import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from torchvision import transforms as T
from torchvision.utils import save_image

import os

# helper functions

def default(val, def_val):
    return def_val if val is None else val

def flatten(t):
    return t.reshape(t.shape[0], -1)

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

def get_module_device(module):
    return next(module.parameters()).device

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

def MaybeSyncBatchnorm(is_distributed = None):
    is_distributed = default(is_distributed, dist.is_initialized() and dist.get_world_size() > 1)
    return nn.SyncBatchNorm if is_distributed else nn.BatchNorm1d

# loss fn

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

# augmentation utils

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

# exponential moving average

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# MLP class for projector and predictor

def MLP(dim, projection_size, hidden_size=4096, sync_batchnorm=None):
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        MaybeSyncBatchnorm(sync_batchnorm)(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )

def SimSiamMLP(dim, projection_size, hidden_size=4096, sync_batchnorm=None):
    return nn.Sequential(
        nn.Linear(dim, hidden_size, bias=False),
        MaybeSyncBatchnorm(sync_batchnorm)(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, hidden_size, bias=False),
        MaybeSyncBatchnorm(sync_batchnorm)(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size, bias=False),
        MaybeSyncBatchnorm(sync_batchnorm)(projection_size, affine=False)
    )

# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets

class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, use_simsiam_mlp = False, sync_batchnorm = None):
        super().__init__()
        self.net = net

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.use_simsiam_mlp = use_simsiam_mlp
        self.sync_batchnorm = sync_batchnorm

    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        create_mlp_fn = MLP if not self.use_simsiam_mlp else SimSiamMLP
        projector = create_mlp_fn(dim, self.projection_size, self.projection_hidden_size, sync_batchnorm = self.sync_batchnorm)
        return projector.to(hidden)

    def get_representation(self, x):
        return self.net(x)

    def forward(self, x, return_projection = True):
        representation = self.get_representation(x)

        if not return_projection:
            return representation

        projector = self._get_projector(representation)
        projection = projector(representation)
        return projection, representation

# main class

class BYOL(nn.Module):
    def __init__(
        self,
        net,
        obs_shape,
        augment_fn,
        augment_fn2,
        add_aug_anchor_and_positive,
        projection_size = 256,
        projection_hidden_size = 4096,
        moving_average_decay = 0.99,
        use_momentum = True,
        sync_batchnorm = None
    ):
        super().__init__()
        self.net = net
        image_size = obs_shape[-1]

        self.augment1 = augment_fn
        self.augment2 = augment_fn2
        self.add_aug_anchor_and_positive = add_aug_anchor_and_positive

        self.online_encoder = NetWrapper(net, 
            projection_size, 
            projection_hidden_size, 
            use_simsiam_mlp = not use_momentum,
            sync_batchnorm = sync_batchnorm
        )

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)

        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)

        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(2, obs_shape[0], image_size, image_size, device=device), 
                     torch.randn(2, obs_shape[0], image_size, image_size, device=device),
                     torch.randn(2, obs_shape[0], image_size, image_size, device=device),
                     torch.randn(2, obs_shape[0], image_size, image_size, device=device),
                     'all')

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def augment(self, x):
        b, c, h, w = x.size()
        assert h == w 

        num_frames = c // 3

        if self.add_aug_anchor_and_positive:
            image_one = []
            image_two = []
            for i in range(num_frames):
                frame = x[:, 3*i:3*i+3, :, :]
                frame_one = self.augment1(frame)
                frame_two = self.augment2(frame)
                image_one.append(frame_one)
                image_two.append(frame_two)

            image_one = torch.cat(image_one, dim=1).float()
            image_two = torch.cat(image_two, dim=1).float()

        else:
            image_two = []
            for i in range(num_frames):
                frame = x[:, 3*i:3*i+3, :, :]
                frame_two = self.augment2(frame)
                image_two.append(frame_two)

            image_two = torch.cat(image_two, dim=1).float()
            image_one = x.float()

        return image_one, image_two

    def forward(self, 
                obs, 
                obs_e_raw, 
                obs_random, 
                obs_e_raw_random, 
                CL_data_type,
                check_every_steps=5000, 
                step=0, 
                return_embedding = False, 
                return_projection = True):
        
        assert not (self.training and obs.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        if CL_data_type == 'agent':
            anchor_to_aug = obs

        elif CL_data_type == 'expert':
            anchor_to_aug = torch.cat([obs, obs_e_raw], dim=0)

        elif CL_data_type == 'all':
            anchor_to_aug = torch.cat([obs, obs_e_raw], dim=0)
            anchors = torch.cat([obs_random, obs_e_raw_random], dim=0)
            rand_idx = torch.randperm(obs_random.shape[0])
            positives = torch.cat([obs_e_raw_random[rand_idx], obs_random[rand_idx]], dim=0)

        elif CL_data_type == 'agent-random':
            anchor_to_aug = obs
            anchors = obs_random
            rand_idx = torch.randperm(obs_random.shape[0])
            positives = torch.cat([obs_e_raw_random[rand_idx], obs_random[rand_idx]], dim=0)

        elif CL_data_type == 'random-only':
            anchors = torch.cat([obs_random, obs_e_raw_random], dim=0)
            rand_idx = torch.randperm(obs_random.shape[0])
            positives = torch.cat([obs_e_raw_random[rand_idx], obs_random[rand_idx]], dim=0)            

        else:
            NotImplementedError
        
        if return_embedding:
            return self.online_encoder(anchor_to_aug, return_projection = return_projection)

        if CL_data_type == 'all' or CL_data_type == 'agent-random':
            image_one, image_two = self.augment(anchor_to_aug)
            anchors = torch.cat([anchors.float(), image_one], dim=0)
            positives = torch.cat([positives.float(), image_two], dim=0)

            if step % check_every_steps == 0:
                self.check_aug(image_one, image_two, step)

        elif CL_data_type == 'random-only':
            anchors = anchors.float()
            positives = positives.float()

        else:
            image_one, image_two = self.augment(anchor_to_aug)
            anchors = image_one
            positives = image_two

            if step % check_every_steps == 0:
                self.check_aug(image_one, image_two, step)

        online_proj_one, _ = self.online_encoder(anchors)
        online_proj_two, _ = self.online_encoder(positives)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            target_proj_one, _ = target_encoder(anchors)
            target_proj_two, _ = target_encoder(positives)
            target_proj_one.detach_()
            target_proj_two.detach_()

        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())

        loss = loss_one + loss_two
        return loss.mean()
    
    def check_aug(self, img1, img2, step):

        if not os.path.exists('checkaugs'):
            os.makedirs("checkaugs")

        img1 = img1/255 # -> [0, 1]
        img2 = img2/255 # -> [0, 1]

        rand_idx = torch.randperm(img1.shape[0])

        imgs1 = img1[rand_idx[:9]]
        imgs2 = img2[rand_idx[:9]]

        imgs3 = img1[rand_idx[-9:]]
        imgs4 = img2[rand_idx[-9:]]
                
        saved_imgs = torch.cat([imgs1[:,:3,:,:], imgs2[:,:3,:,:], imgs3[:,:3,:,:], imgs4[:,:3,:,:]], dim=0)
        save_image(saved_imgs, "./checkaugs/%d.png" % (step), nrow=9)
