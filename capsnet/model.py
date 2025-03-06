import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import json

# CapsNet architecture:
# 1. Sequential Conv layers (Conv2d + ReLU + MaxPool) extract spatial features.
# 2. PrimaryCaps (Conv2d reshaped) maps features to pose vectors, preserving local spatial hierarchies.
# 3. DigitCaps (fully connected capsule layer) applies dynamic routing-by-agreement over num_routes.
# 4. Squash function normalizes capsule vectors for nonlinear representation.
# 5. Capsule activations (vector norms) serve as classification probabilities.
# 6. Masking mechanism zeros non-target capsules to focus reconstruction.
# 7. Decoder (MLP) reconstructs input from activated capsules, enforcing feature retention.
# 8. Routing iteratively refines agreement between PrimaryCaps and DigitCaps.

class PrimaryCapsules(nn.Module):
    def __init__(self, in_channels, num_capsules, capsule_dim, kernel_size, stride):
        super(PrimaryCapsules, self).__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.conv = nn.Conv2d(in_channels, num_capsules * capsule_dim,
                              kernel_size=kernel_size, stride=stride)
        
    def forward(self, x):
        batch_size = x.size(0)
        out = self.conv(x)                                                     # [B, num_capsules * capsule_dim, H, W]
        out = out.view(batch_size, self.num_capsules, self.capsule_dim, -1)
        out = out.transpose(2, 3).contiguous()                                 # [B, num_capsules, N, capsule_dim]
        out = out.view(batch_size, -1, self.capsule_dim)                       # [B, num_capsules * N, capsule_dim]
        return self.squash(out)
    
    def squash(self, s, dim=-1):
        s_norm = torch.norm(s, dim=dim, keepdim=True)
        scale = (s_norm**2) / (1 + s_norm**2)
        v = scale * s / (s_norm + 1e-8)
        return v

class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, in_channels, out_channels, num_routes, routing_iters=1):
        super(CapsuleLayer, self).__init__()
        self.num_capsules = num_capsules
        self.routing_iters = routing_iters
        self.num_routes = num_routes
        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(2).unsqueeze(4)                                        # [B, num_routes, 1, in_channels, 1]
        W = self.W.expand(batch_size, *self.W.size()[1:])
        u_hat = torch.matmul(W, x).squeeze(-1)                                 # [B, num_routes, num_capsules, out_channels]
        
        b = torch.zeros(batch_size, self.num_routes, self.num_capsules, device=x.device)
        for i in range(self.routing_iters):
            c = F.softmax(b, dim=2).unsqueeze(-1)                              # [B, num_routes, num_capsules, 1]
            s = (c * u_hat).sum(dim=1)                                         # [B, num_capsules, out_channels]
            v = self.squash(s)
            if i < self.routing_iters - 1:
                v_expand = v.unsqueeze(1)                                      # [B, 1, num_capsules, out_channels]
                b = b + (u_hat * v_expand).sum(dim=-1)
        return v
    
    def squash(self, s, dim=-1):
        s_norm = torch.norm(s, dim=dim, keepdim=True)
        scale = (s_norm**2) / (1 + s_norm**2)
        v = scale * s / (s_norm + 1e-8)
        return v

import math

class CapsNet(nn.Module):
    def __init__(self,
                 input_size=128,
                 conv_channels=[64, 128, 256],
                 primary_caps_params={'num_capsules': 32, 'capsule_dim': 8, 'kernel_size': 9, 'stride': 2},
                 num_classes=10,
                 capsule_out_dim=16,
                 routing_iters=3,
                 decoder_hidden_dims=[1024, 4096],
                 decoder_dropout_rates=[0.3, 0.3, 0.3]):
        super(CapsNet, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, conv_channels[0], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        size = input_size // 8
        
        self.primary_capsules = PrimaryCapsules(
            in_channels=conv_channels[2],
            num_capsules=primary_caps_params['num_capsules'],
            capsule_dim=primary_caps_params['capsule_dim'],
            kernel_size=primary_caps_params['kernel_size'],
            stride=primary_caps_params['stride']
        )
        
        primary_size = math.floor((size - primary_caps_params['kernel_size']) / primary_caps_params['stride']) + 1
    
        if primary_size <= 0:
            raise ValueError(f"Invalid input size : {primary_size}")

        num_routes = primary_caps_params['num_capsules'] * (primary_size ** 2)
        
        self.digit_capsules = CapsuleLayer(
            num_capsules=num_classes,
            in_channels=primary_caps_params['capsule_dim'],
            out_channels=capsule_out_dim,
            num_routes=num_routes,
            routing_iters=routing_iters
        )
        
        decoder_layers = []
        in_dim = capsule_out_dim * num_classes
        n_hidden = len(decoder_hidden_dims)

        if len(decoder_dropout_rates) != n_hidden + 1:
            raise ValueError("len(decoder_dropout_rates) must be equal to len(decoder_hidden_dims) + 1")

        for i in range(n_hidden):
            decoder_layers.append(nn.Linear(in_dim, decoder_hidden_dims[i]))
            decoder_layers.append(nn.Dropout(decoder_dropout_rates[i]))
            decoder_layers.append(nn.ReLU())
            in_dim = decoder_hidden_dims[i]
            
        decoder_layers.append(nn.Linear(in_dim, input_size * input_size))
        decoder_layers.append(nn.Dropout(decoder_dropout_rates[-1]))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)
        
        self.input_size = input_size
    
    def forward(self, x, y=None):
        x = self.conv_layers(x)
        x = self.primary_capsules(x)
        digit_caps = self.digit_capsules(x)
        probs = torch.norm(digit_caps, dim=-1)
        
        if y is None:
            _, max_indices = probs.max(dim=1)
            y_onehot = F.one_hot(max_indices, digit_caps.size(1)).float()
        else:
            y_onehot = F.one_hot(y, digit_caps.size(1)).float()
        
        masked = (digit_caps * y_onehot.unsqueeze(2)).view(x.size(0), -1)
        reconstruction = self.decoder(masked)
        reconstruction = reconstruction.view(-1, 1, self.input_size, self.input_size)
        return probs, reconstruction

def save_model(model, params, model_name):
    dir = f"models/{model_name}"
    os.makedirs(dir, exist_ok=True)
    torch.save(model.state_dict(), f"{dir}/model.pth")
    with open(f"{dir}/params.json", "w") as f:
        json.dump(params, f)

def load_model(model_name):
    dir = f"models/{model_name}"
    if not os.path.exists(dir):
        raise Exception(f"No model {model_name}")
    
    with open(f"{dir}/params.json", "r") as f:
        params = json.load(f)
    
    model = CapsNet(**params)
    model.load_state_dict(torch.load(f"{dir}/model.pth"))
    
    return model, params