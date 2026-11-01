"""Original EEGPT classifier and evaluation adapters."""
from src.model import *


def evaluate(model, dataloader, criterion, device, use_amp=False, model_name=None, valid_indices=None):
    """Evaluate model on dataloader"""
    model.eval()
    (all_preds, all_labels) = ([], [])
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                (eeg, labels) = batch
            else:
                eeg = batch['eeg']
                labels = batch['label']
            eeg = eeg.to(device)
            labels = labels.to(device)
            if model_name == 'EEGPT':
                if valid_indices is not None:
                    eeg = torch.index_select(eeg, 1, valid_indices)
                if eeg.dim() == 4:
                    eeg = eeg.flatten(2)
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
    return (avg_loss, acc, f1, cm)


def evaluate_multitask(model, dataloader, criterion, device, use_amp=False, model_name=None, valid_indices=None):
    label_mapping = {10: 0, 11: 1, 12: 2, 20: 3, 21: 4, 30: 5, 31: 6, 32: 7, 40: 8, 41: 9, 42: 10}
    model.eval()
    (all_preds, all_labels) = ([], [])
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                (eeg, labels) = batch
            else:
                eeg = batch['eeg']
                labels = batch['label']
                mapped_labels = torch.tensor([label_mapping[label.item()] for label in labels])
            eeg = eeg.to(device)
            mapped_labels = mapped_labels.to(device)
            if model_name == 'EEGPT':
                if valid_indices is not None:
                    eeg = torch.index_select(eeg, 1, valid_indices)
                if eeg.dim() == 4:
                    eeg = eeg.flatten(2)
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
    return (avg_loss, acc, f1, cm)


def evaluate_multitask_cog(model, dataloader, criterion, device, use_amp=False, model_name=None, valid_indices=None):
    label_mapping = {10: 0, 11: 1, 12: 2, 20: 3, 21: 4, 22: 5}
    model.eval()
    (all_preds, all_labels) = ([], [])
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                (eeg, labels) = batch
            else:
                eeg = batch['eeg']
                labels = batch['label']
                mapped_labels = torch.tensor([label_mapping[label.item()] for label in labels])
            eeg = eeg.to(device)
            mapped_labels = mapped_labels.to(device)
            if model_name == 'EEGPT':
                if valid_indices is not None:
                    eeg = torch.index_select(eeg, 1, valid_indices)
                if eeg.dim() == 4:
                    eeg = eeg.flatten(2)
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
    return (avg_loss, acc, f1, cm)


class SimpleClassifier_EEGPT(nn.Module):
    """
    Simple Fine-tuning Classifier for EEGPT
    """

    def __init__(self, encoder, in_chans, args, use_channels_names, num_classes=3, embed_dim=512):
        super().__init__()
        self.encoder = encoder
        self.chans_num = len(use_channels_names)
        self.chans_id = encoder.prepare_chan_ids(use_channels_names)
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        self.model = args.model
        self.classifier_CBM = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(embed_dim, num_classes))
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        Args: x: (B, C, A, T)
        Returns: logits: (B, num_classes)
        """
        features = self.encoder(x, self.chans_id.to(x))
        features = torch.mean(features, dim=[1, 2])
        logits = self.classifier(features)
        return logits
