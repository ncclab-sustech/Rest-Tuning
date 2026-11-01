import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from datetime import datetime
from peft import get_peft_model, LoraConfig
import argparse
import numpy as np
import math
import torch.backends.cudnn as cudnn
import json
import os
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from src.utils import *
from src.cbramod import CBraMod


class Embedding_MaskingStrategy:
    """
    Implements various masking strategies for EEG data (batch, channel, patch, feature).

    Supports:
    - Temporal patch masking: Mask contiguous patches in the patch dimension
    - Channel masking: Mask entire channels
    - Random masking: Random point-wise masking
    - Hybrid masking: Combination of temporal and channel masking
    """

    def __init__(self,
                 mask_ratio: float = 0.4,
                 mask_type: str = 'patch'):
        """
        Args:
            mask_ratio: Proportion of data to mask (0.0 to 1.0)
            mask_type: Type of masking ('temporal_patch', 'channel', 'random', 'hybrid')
            patch_size: Size of temporal patches for patch-based masking
            min_mask_len: Minimum length of masked segments
            max_mask_len: Maximum length of masked segments
        """
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type

    def __call__(self, x: torch.Tensor) -> tuple:
        """
        Apply masking to input EEG data.

        Args:
            x: Input tensor of shape (batch, channels, patches, features)

        Returns:
            masked_x: Masked input (masked regions set to 0)
            mask: Binary mask (1 = masked, 0 = visible)
            original_x: Original unmasked input
        """
        if self.mask_type == 'patch':
            return self._temporal_patch_mask(x)
        elif self.mask_type == 'channel':
            return self._channel_mask(x)
        elif self.mask_type == 'random':
            return self._random_mask(x)
        elif self.mask_type == 'hybrid':
            return self._hybrid_mask(x)
        elif self.mask_type == 'non_mask':
            return x, 0
        else:
            raise ValueError(f"Unknown mask type: {self.mask_type}")

    def _temporal_patch_mask(self, x: torch.Tensor) -> tuple:
        """Mask contiguous temporal patches (in the patch dimension)."""
        batch, channels, patches, features = x.shape
        mask = torch.zeros_like(x)

        n_patches = patches
        n_mask = int(n_patches * self.mask_ratio)

        for b in range(batch):

            mask_indices = torch.randperm(n_patches)[:n_mask]
            for idx in mask_indices:
                start = idx
                end = min(start + 1, patches)
                mask[b, :, start:end, :] = 1

        masked_x = x * (1 - mask)
        return masked_x, mask

    def _channel_mask(self, x: torch.Tensor) -> tuple:
        """Mask entire channels."""
        batch, channels, patches, features = x.shape
        mask = torch.zeros_like(x)

        n_mask_channels = int(channels * self.mask_ratio)

        for b in range(batch):
            mask_channels = torch.randperm(channels)[:n_mask_channels]
            mask[b, mask_channels, :, :] = 1

        masked_x = x * (1 - mask)
        return masked_x, mask

    def _random_mask(self, x: torch.Tensor) -> tuple:
        """Random point-wise masking."""
        batch, channels, patches, features = x.shape
        mask = (torch.rand_like(x) < self.mask_ratio).float()
        masked_x = x * (1 - mask)
        return masked_x, mask

    def _hybrid_mask(self, x: torch.Tensor) -> tuple:
        """Combine temporal and channel masking."""
        batch, channels, patches, features = x.shape
        mask = torch.zeros_like(x)

        n_mask_channels = int(channels * self.mask_ratio / 2)
        for b in range(batch):
            mask_channels = torch.randperm(channels)[:n_mask_channels]
            mask[b, mask_channels, :, :] = 1

        n_mask_patches = int(patches * self.mask_ratio / 2)

        for b in range(batch):
            mask_indices = torch.randperm(patches)[:n_mask_patches]
            for idx in mask_indices:
                start = idx
                end = min(start + 1, patches)

                visible_channels = (mask[b, :, 0, 0] == 0).nonzero(as_tuple=True)[0]
                if len(visible_channels) > 0:
                    mask[b, visible_channels, start:end, :] = 1

        masked_x = x * (1 - mask)
        return masked_x, mask

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def evaluate(model, dataloader, criterion, device, use_amp=False):
    """Evaluate model on dataloader"""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                eeg, labels = batch
            else:
                eeg = batch['eeg']
                labels = batch['label']

            eeg = eeg.to(device)
            labels = labels.to(device)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits = model(eeg)
                    loss = criterion(logits, labels)
            else:
                logits = model(eeg)
                loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    avg_loss = total_loss / len(dataloader)
    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, acc, f1, cm


def evaluate_multitask(model, dataloader, criterion, device, label_mapping, use_amp=False):

    """Evaluate model on dataloader"""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                eeg, labels = batch
            else:
                eeg = batch['eeg']
                labels = batch['label']
                mapped_labels = torch.tensor([label_mapping[label.item()] for label in labels])

            eeg = eeg.to(device)
            mapped_labels = mapped_labels.to(device)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits = model(eeg)
                    loss = criterion(logits, mapped_labels)
            else:
                logits = model(eeg)
                loss = criterion(logits, mapped_labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(mapped_labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    avg_loss = total_loss / len(dataloader)
    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, acc, f1, cm


def evaluate_multitask_cog(model, dataloader, criterion, device, use_amp=False):
    label_mapping = {
        10: 0,
        11: 1,
        12: 2,
        20: 3,
        21: 4,
        22: 5,
    }
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                eeg, labels = batch
            else:
                eeg = batch['eeg']
                labels = batch['label']
                mapped_labels = torch.tensor([label_mapping[label.item()] for label in labels])

            eeg = eeg.to(device)
            mapped_labels = mapped_labels.to(device)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits = model(eeg)
                    loss = criterion(logits, mapped_labels)
            else:
                logits = model(eeg)
                loss = criterion(logits, mapped_labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(mapped_labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    avg_loss = total_loss / len(dataloader)
    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, acc, f1, cm


class SimpleClassifier_multi_model(nn.Module):
    """
    Simple Fine-tuning Classifier (with direct classification after Encoder)

    Architecture:
    1. Encoder (all parameters trainable)
    2. Global Average Pooling (GAP)
    3. Linear Classification Head (1-layer MLP)
    """
    def __init__(self, encoder, in_chans,args, num_channels=61, num_classes=3, embed_dim=256):
        super().__init__()
        self.encoder = encoder
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        self.model = args.model
        self.classifier_CBM = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),

                nn.Linear(embed_dim, num_classes),
            )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        Args: x: (B, C, A, T)
        Returns: logits: (B, num_classes)
        """
        if self.model == 'CBraMod':
            features = self.encoder(x)
            features=rearrange(features,'b c s d -> b d c s')

            logits = self.classifier_CBM(features)

        else:
            features = self.encoder.forward_features(x, input_chans = self.in_chans)
            logits = self.classifier(features)
        return logits


class SimpleClassifier_brainomni(nn.Module):
    """
    Simple Fine-tuning Classifier (with direct classification after Encoder)

    Architecture:
    1. Encoder (all parameters trainable)
    2. Global Average Pooling (GAP)
    3. Linear Classification Head (1-layer MLP)
    """
    def __init__(self, encoder,position,sensortype, args, num_channels=61, num_classes=3, embed_dim=200):
        super().__init__()
        self.encoder = encoder
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.position = position
        self.sensortype=sensortype
        self.model = args.model
        self.classifier_CBM = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),

                nn.Linear(embed_dim, num_classes),
            )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        Args: x: (B, C, A, T)
        Returns: logits: (B, num_classes)
        """
        B,C,A,T = x.shape
        x = x.reshape(B, C, A * T)
        positions = self.position.unsqueeze(0).repeat(B, 1, 1).to(x.device)
        sensor_types = self.sensortype.unsqueeze(0).repeat(B, 1).to(x.device)

        features = self.encoder.encode(x,positions, sensor_types)
        features=rearrange(features,'b c s d -> b d c s')
        logits = self.classifier_CBM(features)

        return logits


class SimpleClassifier_BIOT(nn.Module):
    """
    Simple Fine-tuning Classifier (with direct classification after Encoder)

    Architecture:
    1. Encoder (all parameters trainable)
    2. Global Average Pooling (GAP)
    3. Linear Classification Head (1-layer MLP)
    """
    def __init__(self, encoder, args, ch_names, num_classes=3, embed_dim=256):
        super().__init__()
        self.encoder = encoder
        self.eeg_projector = BIOT_DataProjector()
        self.embed_dim = embed_dim
        self.model = args.model
        self.ch_names =ch_names
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.eeg_projector.channel_project(x, self.ch_names)
        features = self.encoder(x)
        logits = self.classifier(features)

        return logits


class SimpleClassifier(nn.Module):
    """
    Simple Fine-tuning Classifier (with direct classification after Encoder)

    Architecture:
    1. Encoder (all parameters trainable)
    2. Global Average Pooling (GAP)
    3. Linear Classification Head (1-layer MLP)
    """
    def __init__(self, encoder, in_chans,args, num_channels=61, num_classes=3, embed_dim=200):
        super().__init__()
        self.encoder = encoder
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        self.model = args.model
        self.classifier_CBM = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),

                nn.Linear(embed_dim, num_classes),
            )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        Args: x: (B, C, A, T)
        Returns: logits: (B, num_classes)
        """
        if self.model == 'CBraMod':
            features = self.encoder(x)
            features = rearrange(features, 'b c s d -> b d c s')
            logits = self.classifier_CBM(features)
        else:
            features = self.encoder.forward_features(x, input_chans = self.in_chans)
            logits = self.classifier(features)
        return logits


def adjust_alpha(epoch, initial_alpha=0.5, max_alpha=1.0, min_alpha=0.1, alpha_decay_factor=0.01,warmup_epochs=5):

    if epoch < warmup_epochs:
        alpha = initial_alpha
    else:
        alpha = min(initial_alpha + alpha_decay_factor * (epoch-warmup_epochs), max_alpha)
    return alpha


def cosine_annealing_temperature(epoch, initial_temp=0.1, min_temp=0.005, warmup_epochs=5):
    if epoch < warmup_epochs:
        return initial_temp

    progress = (epoch - warmup_epochs) / (300 - warmup_epochs)
    return min_temp + 0.5 * (initial_temp - min_temp) * (1 + math.cos(math.pi * progress))


def label_aware_contrastive_dynamics(mask_emb, truth_emb, labels, epoch, warmup_epoch,
                            temperature=0.07, alpha=0, lambda_mse=0.1, main_weight=1):
    B, D = mask_emb.shape
    device = mask_emb.device

    mask_norm = F.normalize(mask_emb, dim=-1)
    truth_norm = F.normalize(truth_emb, dim=-1)

    current_temperature = cosine_annealing_temperature(epoch,  initial_temp = temperature, min_temp=0.005,warmup_epochs=warmup_epoch )
    alpha = adjust_alpha(epoch, initial_alpha=alpha, max_alpha=0.8,
                         alpha_decay_factor=0.005, warmup_epochs=warmup_epoch)

    sim = torch.matmul(mask_norm, truth_norm.T)
    logits = sim / current_temperature

    label_mat = (labels.unsqueeze(0) == labels.unsqueeze(1))

    targets = torch.zeros((B, B), device=device)

    targets[torch.arange(B), torch.arange(B)] = 1.0

    same_label_mask = label_mat & ~torch.eye(B, device=device, dtype=torch.bool)
    same_label_idx = same_label_mask.nonzero(as_tuple=False)
    targets[same_label_idx[:, 0], same_label_idx[:, 1]] = alpha

    targets = targets / targets.sum(dim=1, keepdim=True)

    assert torch.all(targets >= 0), "Targets contain negative values"
    assert torch.allclose(targets.sum(dim=1), torch.ones(B, device=device)), "Target rows must sum to one"

    loss = F.kl_div(F.log_softmax(logits, dim=-1), targets.detach(), reduction='batchmean')

    loss_mse = F.mse_loss(mask_norm, truth_norm)
    total_loss = main_weight * loss + lambda_mse * loss_mse

    return total_loss, loss_mse, loss


def contrastive_dynamics(mask_emb, truth_emb, labels, epoch, warmup_epoch,
                            temperature=0.07, alpha=0, lambda_mse=0.1,main_weight=1):
    B, D = mask_emb.shape
    device = mask_emb.device

    mask_norm = F.normalize(mask_emb, dim=-1)
    truth_norm = F.normalize(truth_emb, dim=-1)

    current_temperature = cosine_annealing_temperature(epoch,  initial_temp = temperature, min_temp=0.005,warmup_epochs=warmup_epoch )
    alpha =0

    sim = torch.matmul(mask_norm, truth_norm.T)
    logits = sim / current_temperature

    label_mat = (labels.unsqueeze(0) == labels.unsqueeze(1))

    targets = torch.zeros((B, B), device=device)

    targets[torch.arange(B), torch.arange(B)] = 1.0

    same_label_mask = label_mat & ~torch.eye(B, device=device, dtype=torch.bool)
    same_label_idx = same_label_mask.nonzero(as_tuple=False)
    targets[same_label_idx[:, 0], same_label_idx[:, 1]] = alpha

    targets = targets / targets.sum(dim=1, keepdim=True)

    assert torch.all(targets >= 0), "Targets contain negative values"
    assert torch.allclose(targets.sum(dim=1), torch.ones(B, device=device)), "Target rows must sum to one"

    loss = F.kl_div(F.log_softmax(logits, dim=-1), targets.detach(), reduction='batchmean')

    loss_mse = F.mse_loss(mask_norm, truth_norm)
    total_loss = main_weight * loss + lambda_mse * loss_mse

    return total_loss, loss_mse, loss


def contrastive_4d_channel_dynamics(subject_embeddings, group_embeddings, labels, args,
                                    epoch, warmup_epoch=5, temperature=0.07, alpha=0, lambda_mse=0.1, main_weight=1,
                                    batch_size=None, channel=None, patch=None, dimension=None):
    B = batch_size
    C= channel
    A = patch
    D = dimension
    if args.model == 'labram_base_patch200_200':
        subject_embeddings = rearrange(subject_embeddings, 'b (c a) t -> b c a t', c=C, a=A, t=D)
        group_embeddings = rearrange(group_embeddings, 'b (c a) t -> b c a t', c=C, a=A, t=D)
        B, C, A, D = subject_embeddings.shape
        subject_flat = subject_embeddings.view(B, C, -1)
        group_flat = group_embeddings.view(B, C, -1)
    elif args.model in ('CBraMod', 'EEGPT'):
        B, C, A, D = subject_embeddings.shape
        subject_flat = subject_embeddings.view(B, C, -1)
        group_flat = group_embeddings.view(B, C, -1)

    device = subject_embeddings.device

    channel_losses = []
    channel_intra_losses = []
    channel_cross_losses = []

    for c in range(C):

        subject_c = group_flat[:, c, :]
        group_c = subject_flat[:, c, :]

        loss_c, intra_c, cross_c = label_aware_contrastive_dynamics(
            subject_c, group_c, labels, epoch, warmup_epoch, temperature, alpha, lambda_mse, main_weight
        )

        channel_losses.append(loss_c)
        channel_intra_losses.append(intra_c)
        channel_cross_losses.append(cross_c)

    total_loss = torch.stack(channel_losses).mean()
    avg_intra = torch.stack([l for l in channel_intra_losses if torch.is_tensor(l)]).mean() if channel_intra_losses else torch.tensor(0.0, device=device)
    avg_cross = torch.stack(channel_cross_losses).mean()

    return total_loss, avg_intra, avg_cross
