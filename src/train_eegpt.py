"""EEGPT training functions from the original personalization experiments."""
from functools import partial
import torch.nn.utils as utils
from src.train import *
from src.eegpt import EEGTransformer, CHANNEL_DICT
from src.model_eegpt import *


def train_single_subject_cls_new(args, subject_id, device, num_patches, dataset):
    """Train a subject classifier with the selected task adaptation method"""
    try:
        valid_fitune_types = ['lora', 'full']
        valid_cls_fitune_types = ['lora', 'full', 'linear']
        if args.fitune_type not in valid_fitune_types:
            raise ValueError(f'fitune_type must be one of {valid_fitune_types}')
        if args.cls_fitune_type not in valid_cls_fitune_types:
            raise ValueError(f'cls_fitune_type must be one of {valid_cls_fitune_types}')
        output_path = os.path.join(args.output_root, args.dataset, args.cls_fitune_type)
        subject_dir = os.path.join(output_path, f'subject_{subject_id:02d}')
        os.makedirs(subject_dir, exist_ok=True)
        metrics_path = os.path.join(subject_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            print(f'  ✓ Subject {subject_id:02d} already trained, skipping...')
            with open(metrics_path, 'r') as f:
                return json.load(f)
        print(f"\n{'=' * 80}")
        print(f'Subject {subject_id:02d} - Loading Data (Cognitive Task)')
        print(f"{'=' * 80}")
        if args.fewshot:
            (train_loader, val_loader, test_loader) = create_single_subject_few_shot_dataloaders(dataset=dataset, target_subject_id=subject_id, n_shot=args.n_shot, batch_size=args.batch_size, random_state=args.seed, num_workers=args.num_workers, val_size=args.val_size, oversample=args.oversample)
        elif args.dataset_suffle:
            (train_loader, val_loader, test_loader) = create_single_subject_dataloaders(dataset=dataset, target_subject_id=subject_id, train_size=args.train_size, test_size=args.test_size, val_size=args.val_size, batch_size=args.batch_size, random_state=args.seed, num_workers=args.num_workers)
        else:
            (train_loader, val_loader, test_loader) = create_single_subject_dataloaders_order(dataset=dataset, target_subject_id=subject_id, train_size=args.train_size, test_size=args.test_size, val_size=args.val_size, batch_size=args.batch_size, random_state=args.seed, num_workers=args.num_workers)
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)
        print(f'  Train: {train_size}, Val: {val_size}, Test: {test_size}')
        if train_size < args.min_samples_per_class:
            raise ValueError(f'Insufficient training samples: {train_size}')
        if args.dataset == 'cogbci':
            (in_chans, ch_names, chan_number) = get_cogbci_dataset(args, subject_id)
            print('in_chans', in_chans)
        else:
            (in_chans, ch_names, chan_number) = get_dataset_name(args)
            print('in_chans', in_chans)
        valid_indices = None
        from src.eegpt import CHANNEL_DICT
        valid_indices_list = []
        valid_ch_names = []
        for (i, name) in enumerate(ch_names):
            clean_name = name.upper().strip('.')
            if clean_name in CHANNEL_DICT:
                valid_indices_list.append(i)
                valid_ch_names.append(name)
            else:
                print(f"  Skipping channel '{name}': not in the EEGPT channel dictionary.")
        if len(valid_indices_list) == 0:
            raise ValueError('No valid EEGPT channels remain; check the channel names.')
        ch_names = valid_ch_names
        valid_indices = torch.tensor(valid_indices_list).long().to(device)
        print(f'  Retained channels: {len(ch_names)}')
        checkpoint_path = os.path.join(args.recon_results_dir, args.dataset, f'subject_{subject_id:02d}', 'best_model.pth')
        print(f'  Checkpoint: {checkpoint_path}')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f'Pretrained encoder checkpoint not found: {checkpoint_path}')
        print(f'\n[Step 1] Creating Encoder ({args.model})...')
        from functools import partial
        current_num_channels = len(ch_names)
        print('current_num_channels', current_num_channels)
        encoder = EEGTransformer(img_size=[current_num_channels, 200], patch_size=64, patch_stride=64, embed_num=4, embed_dim=512, depth=8, num_heads=8, mlp_ratio=4.0, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, init_std=0.02, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-06)).to(device)
        if args.fitune_type == 'lora':
            encoder = create_lora(encoder, args)
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            if args.cls_fitune_type == 'lora':
                for (name, param) in encoder.named_parameters():
                    if 'lora' not in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            elif args.cls_fitune_type == 'full':
                for (name, param) in encoder.named_parameters():
                    if param.is_floating_point():
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            elif args.cls_fitune_type == 'linear':
                for (name, param) in encoder.named_parameters():
                    param.requires_grad = False
        elif args.fitune_type == 'full':
            encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
            if args.cls_fitune_type == 'lora':
                encoder = create_lora_cls(encoder, args)
                for (name, param) in encoder.named_parameters():
                    if 'lora' not in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            elif args.cls_fitune_type == 'full':
                for (name, param) in encoder.named_parameters():
                    if param.is_floating_point():
                        param.requires_grad = True
            elif args.cls_fitune_type == 'linear':
                for (name, param) in encoder.named_parameters():
                    param.requires_grad = False
        model = SimpleClassifier_EEGPT(encoder=encoder, in_chans=in_chans, args=args, use_channels_names=ch_names, num_classes=args.num_classes).to(device)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        from torch.cuda.amp import GradScaler
        scaler = GradScaler() if args.use_amp else None
        best_val_acc = float("-inf")
        best_val_f1 = float("-inf")
        patience_counter = 0
        print(f"\n{'=' * 80}")
        print(f'Subject {subject_id:02d} - Training task adaptation')
        print(f'Method: {args.cls_fitune_type.upper()}')
        print(f"{'=' * 80}")
        for epoch in range(args.epochs):
            if epoch < args.warmup_epochs:
                lr_mult = (epoch + 1) / args.warmup_epochs
                current_lr = args.lr * lr_mult
            else:
                progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
                current_lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            model.train()
            train_loss = 0
            for batch in train_loader:
                if isinstance(batch, (tuple, list)):
                    (eeg, labels) = batch
                else:
                    eeg = batch['eeg']
                    labels = batch['label']
                eeg = eeg.to(device)
                labels = labels.to(device)
                if valid_indices is not None:
                    eeg = torch.index_select(eeg, 1, valid_indices)
                if eeg.dim() == 4:
                    eeg = eeg.flatten(2)
                optimizer.zero_grad()
                if args.use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = model(eeg)
                        loss = criterion(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(eeg)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)
            (val_loss, val_acc, val_f1, val_cm) = evaluate(model, val_loader, criterion, device, args.use_amp, model_name=args.model, valid_indices=valid_indices)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f'  Epoch {epoch + 1:3d}/{args.epochs}: TrainLoss={avg_train_loss:.4f}, ValAcc={val_acc:.4f}, ValF1={val_f1:.4f}, LR={current_lr:.6f}')
            if val_f1 > best_val_f1:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                patience_counter = 0
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_acc': val_acc, 'val_f1': val_f1}, os.path.join(subject_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f'  Early stopping at epoch {epoch + 1}')
                    break
        print(f"\n{'=' * 80}")
        print(f'Subject {subject_id:02d} - Final Test Evaluation')
        print(f"{'=' * 80}")
        checkpoint = torch.load(os.path.join(subject_dir, 'best_model.pth'), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        (test_loss, test_acc, test_f1, test_cm) = evaluate(model, test_loader, criterion, device, args.use_amp, model_name=args.model, valid_indices=valid_indices)
        print(f'  Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}')
        print(f'  Confusion Matrix:\n{test_cm}')
        metrics = {'subject_id': int(subject_id), 'method': args.cls_fitune_type, 'train_size': train_size, 'val_size': val_size, 'test_size': test_size, 'best_val_acc': float(best_val_acc), 'best_val_f1': float(best_val_f1), 'test_acc': float(test_acc), 'test_f1': float(test_f1), 'test_cm': test_cm.tolist(), 'best_epoch': int(checkpoint['epoch']) + 1, 'transfer_method': 'linear_probing', 'source_task': 'resting_reconstruction', 'target_task': 'cognitive_classification'}
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f'  ✓ Results saved to {subject_dir}')
        del model
        del optimizer
        del encoder
        torch.cuda.empty_cache()
        return metrics
    except Exception as e:
        print(f'  ✗ ERROR for Subject {subject_id:02d}: {str(e)}')
        import traceback
        traceback.print_exc()
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()
        return None


def train_single_subject_cls_multi_task(args, subject_id, device, num_patches, dataset):
    """Train a subject classifier with the selected task adaptation method"""
    label_mapping = {10: 0, 11: 1, 12: 2, 20: 3, 21: 4, 30: 5, 31: 6, 32: 7, 40: 8, 41: 9, 42: 10}
    try:
        valid_fitune_types = ['lora', 'full']
        valid_cls_fitune_types = ['lora', 'full', 'linear']
        if args.fitune_type not in valid_fitune_types:
            raise ValueError(f'fitune_type must be one of {valid_fitune_types}')
        if args.cls_fitune_type not in valid_cls_fitune_types:
            raise ValueError(f'cls_fitune_type must be one of {valid_cls_fitune_types}')
        output_path = os.path.join(args.output_root, args.dataset, args.cls_fitune_type)
        subject_dir = os.path.join(output_path, f'subject_{subject_id:02d}')
        os.makedirs(subject_dir, exist_ok=True)
        metrics_path = os.path.join(subject_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            print(f'  ✓ Subject {subject_id:02d} already trained, skipping...')
            with open(metrics_path, 'r') as f:
                return json.load(f)
        print(f"\n{'=' * 80}")
        print(f'Subject {subject_id:02d} - Loading Data')
        print(f"{'=' * 80}")
        if args.multi_task:
            (train_loader, val_loader, test_loader) = create_multitask_single_subject_dataloaders(dataset=dataset, target_subject_id=subject_id, target_task_id=[1, 2, 3, 4], train_size=args.train_size, test_size=args.test_size, val_size=args.val_size, batch_size=args.batch_size, random_state=args.seed, num_workers=args.num_workers)
        elif args.multitask_fewshot:
            (train_loader, val_loader, test_loader) = create_multitask_single_subject_few_shot_dataloaders(dataset=dataset, target_subject_id=subject_id, target_task_id=[1, 2, 3, 4], n_shot=args.n_shot, batch_size=args.batch_size, random_state=args.seed, num_workers=args.num_workers, val_size=args.val_size, oversample=args.oversample)
        elif args.fewshot:
            (train_loader, val_loader, test_loader) = create_single_subject_few_shot_dataloaders(dataset=dataset, target_subject_id=subject_id, n_shot=args.n_shot, batch_size=args.batch_size, random_state=args.seed, num_workers=args.num_workers, val_size=args.val_size, oversample=args.oversample)
        elif args.dataset_suffle:
            (train_loader, val_loader, test_loader) = create_single_subject_dataloaders(dataset=dataset, target_subject_id=subject_id, train_size=args.train_size, test_size=args.test_size, val_size=args.val_size, batch_size=args.batch_size, random_state=args.seed, num_workers=args.num_workers)
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)
        print(f'  Train: {train_size}, Val: {val_size}, Test: {test_size}')
        if train_size < args.min_samples_per_class:
            raise ValueError(f'Insufficient training samples: {train_size}')
        if args.dataset == 'cogbci':
            (in_chans, ch_names, chan_number) = get_cogbci_dataset(args, subject_id)
            print('in_chans', in_chans)
        else:
            (in_chans, ch_names, chan_number) = get_dataset_name(args)
            print('in_chans', in_chans)
        valid_indices = None
        from src.eegpt import CHANNEL_DICT
        valid_indices_list = []
        valid_ch_names = []
        for (i, name) in enumerate(ch_names):
            clean_name = name.upper().strip('.')
            if clean_name in CHANNEL_DICT:
                valid_indices_list.append(i)
                valid_ch_names.append(name)
            else:
                print(f"  Skipping channel '{name}': not in the EEGPT channel dictionary.")
        if len(valid_indices_list) == 0:
            raise ValueError('No valid EEGPT channels remain; check the channel names.')
        ch_names = valid_ch_names
        valid_indices = torch.tensor(valid_indices_list).long().to(device)
        print(f'  Retained channels: {len(ch_names)}')
        checkpoint_path = os.path.join(args.recon_results_dir, args.dataset, f'subject_{subject_id:02d}', 'best_model.pth')
        print(f'  Checkpoint: {checkpoint_path}')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f'Pretrained encoder checkpoint not found: {checkpoint_path}')
        print(f'\n[Step 1] Creating Encoder ({args.model})...')
        from functools import partial
        current_num_channels = len(ch_names)
        print('current_num_channels', current_num_channels)
        encoder = EEGTransformer(img_size=[current_num_channels, 200], patch_size=64, patch_stride=64, embed_num=4, embed_dim=512, depth=8, num_heads=8, mlp_ratio=4.0, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, init_std=0.02, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-06)).to(device)
        if args.fitune_type == 'lora':
            encoder = create_lora(encoder, args)
            checkpoint = torch.load(checkpoint_path)
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            if args.cls_fitune_type == 'lora':
                for (name, param) in encoder.named_parameters():
                    if 'lora' not in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            elif args.cls_fitune_type == 'full':
                for (name, param) in encoder.named_parameters():
                    if param.is_floating_point():
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            elif args.cls_fitune_type == 'linear':
                for (name, param) in encoder.named_parameters():
                    param.requires_grad = False
        elif args.fitune_type == 'full':
            encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
            if args.cls_fitune_type == 'lora':
                encoder = create_lora_cls(encoder, args)
                for (name, param) in encoder.named_parameters():
                    if 'lora' not in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            elif args.cls_fitune_type == 'full':
                for (name, param) in encoder.named_parameters():
                    if param.is_floating_point():
                        param.requires_grad = True
            elif args.cls_fitune_type == 'linear':
                for (name, param) in encoder.named_parameters():
                    param.requires_grad = False
        model = SimpleClassifier_EEGPT(encoder=encoder, in_chans=in_chans, args=args, use_channels_names=ch_names, num_classes=args.num_classes).to(device)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        from torch.cuda.amp import GradScaler
        scaler = GradScaler() if args.use_amp else None
        best_val_acc = float("-inf")
        best_val_f1 = float("-inf")
        patience_counter = 0
        print(f"\n{'=' * 80}")
        print(f'Subject {subject_id:02d} - Training task adaptation')
        print(f'Method: {args.cls_fitune_type.upper()}')
        print(f"{'=' * 80}")
        for epoch in range(args.epochs):
            if epoch < args.warmup_epochs:
                lr_mult = (epoch + 1) / args.warmup_epochs
                current_lr = args.lr * lr_mult
            else:
                progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
                current_lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            model.train()
            train_loss = 0
            for batch in train_loader:
                if isinstance(batch, (tuple, list)):
                    (eeg, labels) = batch
                else:
                    eeg = batch['eeg']
                    labels = batch['label']
                    mapped_labels = torch.tensor([label_mapping[label.item()] for label in labels])
                eeg = eeg.to(device)
                mapped_labels = mapped_labels.to(device)
                if valid_indices is not None:
                    eeg = torch.index_select(eeg, 1, valid_indices)
                if eeg.dim() == 4:
                    eeg = eeg.flatten(2)
                optimizer.zero_grad()
                if args.use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = model(eeg)
                        loss = criterion(logits, mapped_labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(eeg)
                    loss = criterion(logits, mapped_labels)
                    loss.backward()
                    optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)
            (val_loss, val_acc, val_f1, val_cm) = evaluate_multitask(model, val_loader, criterion, device, args.use_amp, model_name=args.model, valid_indices=valid_indices)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f'  Epoch {epoch + 1:3d}/{args.epochs}: TrainLoss={avg_train_loss:.4f}, ValAcc={val_acc:.4f}, ValF1={val_f1:.4f}, LR={current_lr:.6f}')
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                patience_counter = 0
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_acc': val_acc, 'val_f1': val_f1}, os.path.join(subject_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f'  Early stopping at epoch {epoch + 1}')
                    break
        print(f"\n{'=' * 80}")
        print(f'Subject {subject_id:02d} - Final Test Evaluation')
        print(f"{'=' * 80}")
        checkpoint = torch.load(os.path.join(subject_dir, 'best_model.pth'), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        (test_loss, test_acc, test_f1, test_cm) = evaluate_multitask(model, test_loader, criterion, device, args.use_amp, model_name=args.model, valid_indices=valid_indices)
        print(f'  Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}')
        print(f'  Confusion Matrix:\n{test_cm}')
        metrics = {'subject_id': int(subject_id), 'method': args.cls_fitune_type, 'train_size': train_size, 'val_size': val_size, 'test_size': test_size, 'best_val_acc': float(best_val_acc), 'best_val_f1': float(best_val_f1), 'test_acc': float(test_acc), 'test_f1': float(test_f1), 'test_cm': test_cm.tolist(), 'best_epoch': int(checkpoint['epoch']) + 1, 'transfer_method': 'linear_probing', 'source_task': 'resting_reconstruction', 'target_task': 'cognitive_classification'}
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f'  ✓ Results saved to {subject_dir}')
        del model
        del optimizer
        del encoder
        torch.cuda.empty_cache()
        return metrics
    except Exception as e:
        print(f'  ✗ ERROR for Subject {subject_id:02d}: {str(e)}')
        import traceback
        traceback.print_exc()
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()
        return None


def train_single_subject_baseline_cls_new(args, subject_id, device, num_patches, dataset):
    """Train a subject classifier with the selected task adaptation method"""
    try:
        output_path = os.path.join(args.output_root, args.dataset, args.cls_fitune_type)
        subject_dir = os.path.join(output_path, f'subject_{subject_id:02d}')
        os.makedirs(subject_dir, exist_ok=True)
        metrics_path = os.path.join(subject_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            print(f'  ✓ Subject {subject_id:02d} already trained, skipping...')
            with open(metrics_path, 'r') as f:
                return json.load(f)
        print(f"\n{'=' * 80}")
        print(f'Subject {subject_id:02d} - Loading Data (Cognitive Task)')
        print(f"{'=' * 80}")
        if args.fewshot:
            (train_loader, val_loader, test_loader) = create_single_subject_few_shot_dataloaders(dataset=dataset, target_subject_id=subject_id, n_shot=args.n_shot, batch_size=args.batch_size, random_state=args.seed, num_workers=args.num_workers, val_size=args.val_size, oversample=args.oversample)
        elif args.dataset_suffle:
            (train_loader, val_loader, test_loader) = create_single_subject_dataloaders(dataset=dataset, target_subject_id=subject_id, train_size=args.train_size, test_size=args.test_size, val_size=args.val_size, batch_size=args.batch_size, random_state=args.seed, num_workers=args.num_workers)
        else:
            (train_loader, val_loader, test_loader) = create_single_subject_dataloaders_order(dataset=dataset, target_subject_id=subject_id, train_size=args.train_size, test_size=args.test_size, val_size=args.val_size, batch_size=args.batch_size, random_state=args.seed, num_workers=args.num_workers)
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)
        print(f'  Train: {train_size}, Val: {val_size}, Test: {test_size}')
        if train_size < args.min_samples_per_class:
            raise ValueError(f'Insufficient training samples: {train_size}')
        if args.dataset == 'cogbci':
            (in_chans, ch_names, chan_number) = get_cogbci_dataset(args, subject_id)
            print('in_chans', in_chans)
        else:
            (in_chans, ch_names, chan_number) = get_dataset_name(args)
            print('in_chans', in_chans)
        valid_indices = None
        from src.eegpt import CHANNEL_DICT
        valid_indices_list = []
        valid_ch_names = []
        for (i, name) in enumerate(ch_names):
            clean_name = name.upper().strip('.')
            if clean_name in CHANNEL_DICT:
                valid_indices_list.append(i)
                valid_ch_names.append(name)
            else:
                print(f"  Skipping channel '{name}': not in the EEGPT channel dictionary.")
        if len(valid_indices_list) == 0:
            raise ValueError('No valid EEGPT channels remain; check the channel names.')
        ch_names = valid_ch_names
        valid_indices = torch.tensor(valid_indices_list).long().to(device)
        print(f'  Retained channels: {len(ch_names)}')
        print(f'\n[Step 1] Creating Encoder ({args.model})...')
        from functools import partial
        current_num_channels = len(ch_names)
        print('current_num_channels', current_num_channels)
        encoder = EEGTransformer(img_size=[current_num_channels, 200], patch_size=64, patch_stride=64, embed_num=4, embed_dim=512, depth=8, num_heads=8, mlp_ratio=4.0, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, init_std=0.02, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-06)).to(device)
        pretrain_ckpt = torch.load(args.pretrained_path, weights_only=False)
        target_encoder_stat = {}
        for (k, v) in pretrain_ckpt['state_dict'].items():
            if k.startswith('target_encoder.'):
                target_encoder_stat[k[15:]] = v
        encoder.load_state_dict(target_encoder_stat)
        if args.fitune_type == 'lora':
            encoder = create_lora(encoder, args)
            for (name, param) in encoder.named_parameters():
                if 'lora' not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        else:
            for (name, param) in encoder.named_parameters():
                param.requires_grad = True
        if args.cls_fitune_type == 'lora':
            encoder = create_lora(encoder, args)
            for (name, param) in encoder.named_parameters():
                if 'lora' not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        elif args.cls_fitune_type == 'full':
            for (name, param) in encoder.named_parameters():
                param.requires_grad = True
        elif args.cls_fitune_type == 'linear':
            for (name, param) in encoder.named_parameters():
                param.requires_grad = False
        model = SimpleClassifier_EEGPT(encoder=encoder, in_chans=in_chans, args=args, use_channels_names=ch_names, num_classes=args.num_classes).to(device)
        trainable = [p for p in encoder.parameters() if p.requires_grad]
        total = sum((p.numel() for p in encoder.parameters()))
        trainable_count = sum((p.numel() for p in trainable))
        print(f'trainable: {trainable_count:,} / {total:,}')
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        from torch.cuda.amp import GradScaler
        scaler = GradScaler() if args.use_amp else None
        best_val_acc = float("-inf")
        best_val_f1 = float("-inf")
        patience_counter = 0
        print(f"\n{'=' * 80}")
        print(f'Subject {subject_id:02d} - Training task adaptation')
        print(f'Method: {args.cls_fitune_type.upper()}')
        print(f"{'=' * 80}")
        for epoch in range(args.epochs):
            if epoch < args.warmup_epochs:
                lr_mult = (epoch + 1) / args.warmup_epochs
                current_lr = args.lr * lr_mult
            else:
                progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
                current_lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            model.train()
            train_loss = 0
            for batch in train_loader:
                if isinstance(batch, (tuple, list)):
                    (eeg, labels) = batch
                else:
                    eeg = batch['eeg']
                    labels = batch['label']
                eeg = eeg.to(device)
                labels = labels.to(device)
                if valid_indices is not None:
                    eeg = torch.index_select(eeg, 1, valid_indices)
                if eeg.dim() == 4:
                    eeg = eeg.flatten(2)
                optimizer.zero_grad()
                if args.use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = model(eeg)
                        loss = criterion(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(eeg)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)
            (val_loss, val_acc, val_f1, val_cm) = evaluate(model, val_loader, criterion, device, args.use_amp, model_name=args.model, valid_indices=valid_indices)
            if epoch == 0:
                torch.save({'model_state_dict': model.state_dict()}, os.path.join(subject_dir, 'original_model.pth'))
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f'  Epoch {epoch + 1:3d}/{args.epochs}: TrainLoss={avg_train_loss:.4f}, ValAcc={val_acc:.4f}, ValF1={val_f1:.4f}, LR={current_lr:.6f}')
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                patience_counter = 0
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'classifier_state_dict': model.classifier.state_dict(), 'val_acc': val_acc, 'val_f1': val_f1}, os.path.join(subject_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f'  Early stopping at epoch {epoch + 1}')
                    break
        print(f"\n{'=' * 80}")
        print(f'Subject {subject_id:02d} - Final Test Evaluation')
        print(f"{'=' * 80}")
        checkpoint = torch.load(os.path.join(subject_dir, 'best_model.pth'), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        (test_loss, test_acc, test_f1, test_cm) = evaluate(model, test_loader, criterion, device, args.use_amp, model_name=args.model, valid_indices=valid_indices)
        print(f'  Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}')
        print(f'  Confusion Matrix:\n{test_cm}')
        metrics = {'subject_id': int(subject_id), 'method': args.cls_fitune_type, 'train_size': train_size, 'val_size': val_size, 'test_size': test_size, 'best_val_acc': float(best_val_acc), 'best_val_f1': float(best_val_f1), 'test_acc': float(test_acc), 'test_f1': float(test_f1), 'test_cm': test_cm.tolist(), 'best_epoch': int(checkpoint['epoch']) + 1, 'transfer_method': 'linear_probing', 'source_task': 'resting_reconstruction', 'target_task': 'cognitive_classification'}
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f'  ✓ Results saved to {subject_dir}')
        del model
        del optimizer
        del encoder
        torch.cuda.empty_cache()
        return metrics
    except Exception as e:
        print(f'  ✗ ERROR for Subject {subject_id:02d}: {str(e)}')
        import traceback
        traceback.print_exc()
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()
        return None


def train_for_subject_embedding_with_dynamics(args, subject_id, device, num_patches, dataset, encoder_group):
    """Calibrate a subject encoder against the fixed population teacher."""
    try:
        subject_dir = os.path.join(args.output_root, args.dataset, f'subject_{subject_id:02d}')
        os.makedirs(subject_dir, exist_ok=True)
        print(f"\n{'=' * 80}")
        print(f'Subject {subject_id:02d} - Loading Data')
        print(f"{'=' * 80}")
        if args.multi_task:
            (train_loader, val_loader, test_loader) = create_multitask_single_subject_dataloaders(dataset=dataset, target_subject_id=subject_id, target_task_id=args.multi_task_ids, train_size=args.train_size, test_size=args.test_size, val_size=args.val_size, batch_size=args.batch_size, random_state=args.seed, num_workers=args.num_workers)
        elif args.train_count is not None:
            (train_loader, val_loader, test_loader) = create_single_subject_dataloaders_control(dataset=dataset, target_subject_id=subject_id, train_size=args.train_size, test_size=args.test_size, val_size=args.val_size, batch_size=args.batch_size, random_state=args.seed, num_workers=args.num_workers, train_count=args.train_count)
        elif args.dataset_suffle:
            (train_loader, val_loader, test_loader) = create_single_subject_dataloaders(dataset=dataset, target_subject_id=subject_id, train_size=args.train_size, test_size=args.test_size, val_size=args.val_size, batch_size=args.batch_size, random_state=args.seed, num_workers=args.num_workers)
        else:
            (train_loader, val_loader, test_loader) = create_single_subject_dataloaders_order(dataset=dataset, target_subject_id=subject_id, train_size=args.train_size, test_size=args.test_size, val_size=args.val_size, batch_size=args.batch_size, random_state=args.seed, num_workers=args.num_workers)
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)
        print(f'  Train: {train_size}, Val: {val_size}, Test: {test_size}')
        if train_size < args.min_samples_per_class:
            raise ValueError(f'Insufficient training samples: {train_size}')
        if test_size == 0:
            print('Warning: Test set is empty! Skipping test evaluation.')
            test_metrics = None
        else:
            test_metrics = True
        if args.dataset == 'cogbci':
            (in_chans, ch_names, chan_number) = get_cogbci_dataset(args, subject_id)
            print('in_chans', in_chans)
        else:
            (in_chans, ch_names, chan_number) = get_dataset_name(args)
            print('in_chans', in_chans)
        valid_indices = None
        from src.eegpt import CHANNEL_DICT
        valid_indices_list = []
        valid_ch_names = []
        for (i, name) in enumerate(ch_names):
            clean_name = name.upper().strip('.')
            if clean_name in CHANNEL_DICT:
                valid_indices_list.append(i)
                valid_ch_names.append(name)
            else:
                print(f"  Skipping channel '{name}': not in the EEGPT channel dictionary.")
        if len(valid_indices_list) == 0:
            raise ValueError('No valid EEGPT channels remain; check the channel names.')
        ch_names = valid_ch_names
        valid_indices = torch.tensor(valid_indices_list).long().to(device)
        print(f'  Retained channels: {len(ch_names)}')
        for (name, param) in encoder_group.named_parameters():
            param.requires_grad = False
        from functools import partial
        current_num_channels = len(ch_names)
        print('current_num_channels', current_num_channels)
        encoder_sub = EEGTransformer(img_size=[current_num_channels, 200], patch_size=64, patch_stride=64, embed_num=4, embed_dim=512, depth=8, num_heads=8, mlp_ratio=4.0, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, init_std=0.02, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-06)).to(device)
        pretrain_ckpt = torch.load(args.pretrained_path, map_location='cpu', weights_only=False)
        target_encoder_stat = {}
        for (k, v) in pretrain_ckpt['state_dict'].items():
            if k.startswith('target_encoder.'):
                target_encoder_stat[k[15:]] = v
        encoder_sub.load_state_dict(target_encoder_stat)
        if args.fitune_type == 'lora':
            encoder_sub = create_lora(encoder_sub, args)
            for (name, param) in encoder_sub.named_parameters():
                if 'lora' not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        else:
            for (name, param) in encoder_sub.named_parameters():
                param.requires_grad = True
        (total_params, trainable_params, trainable_percentage, trainable_modules) = calculate_parameter_percentage_with_names(encoder_sub)
        print(f'encoder_sub Total parameters: {total_params}')
        print(f'encoder_sub Trainable parameters: {trainable_params}')
        print(f'encoder_sub Trainable parameters percentage: {trainable_percentage:.2f}%')
        trainable_params = [p for p in encoder_sub.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        print(f"\n{'=' * 80}")
        print(f'Subject {subject_id:02d} - Training ({args.fitune_type})')
        print(f"{'=' * 80}")
        masking_strategy = Embedding_MaskingStrategy(mask_ratio=args.mask_ratio, mask_type=args.mask_type)
        max_grad_norm = 1.0
        for epoch in range(args.epochs):
            if epoch < args.warmup_epochs:
                lr_mult = (epoch + 1) / args.warmup_epochs
                current_lr = args.lr * lr_mult
            else:
                progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
                current_lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            encoder_sub.train()
            encoder_group.eval()
            train_loss = 0
            train_loss_intra = 0
            train_loss_cross = 0
            for batch in train_loader:
                optimizer.zero_grad()
                eeg = batch['eeg'].to(device)
                label = batch['label'].to(device)
                if args.model == 'EEGPT' and valid_indices is not None:
                    eeg = torch.index_select(eeg, 1, valid_indices)
                (B, C, A, T) = eeg.shape
                if args.embedding_high_level:
                    (mask_eeg, mask) = masking_strategy(eeg)
                    if eeg.dim() == 4:
                        eeg = eeg.flatten(2)
                    if mask_eeg.dim() == 4:
                        mask_eeg = mask_eeg.flatten(2)
                    chans_id = encoder_sub.prepare_chan_ids(ch_names)
                    chans_id = chans_id.to(device)
                    embedding_mask = encoder_sub(mask_eeg, chans_id.to(mask_eeg))
                    embedding_truth = encoder_group(eeg, chans_id.to(eeg))
                    (loss, loss_intra, loss_cross) = contrastive_4d_channel_dynamics(embedding_mask, embedding_truth, label, args, epoch=epoch, warmup_epoch=args.warmup_epochs, temperature=args.contrastive_task_temperature, alpha=args.contrastive_task_alpha, main_weight=args.main_weight, lambda_mse=args.lambda_mse, batch_size=B, channel=C, patch=A, dimension=T)
                else:
                    (loss, loss_intra, loss_cross) = contrastive_dynamics(embedding_mask, embedding_truth, label, epoch=epoch, warmup_epoch=args.warmup_epochs, main_weight=args.main_weight, lambda_mse=args.lambda_mse, temperature=args.contrastive_task_temperature, alpha=args.contrastive_task_alpha)
                loss.backward()
                utils.clip_grad_norm_(encoder_sub.parameters(), max_grad_norm)
                optimizer.step()
                train_loss += loss.item()
                train_loss_intra += loss_intra.item()
                train_loss_cross += loss_cross.item()
            avg_train_loss = train_loss / len(train_loader)
            avg_train_loss_intra = train_loss_intra / len(train_loader)
            avg_train_loss_cross = train_loss_cross / len(train_loader)
            train_losses.append(avg_train_loss)
            encoder_sub.eval()
            encoder_group.eval()
            val_loss = 0
            val_loss_intra = 0
            val_loss_cross = 0
            with torch.no_grad():
                for batch in val_loader:
                    eeg = batch['eeg'].to(device)
                    label = batch['label'].to(device)
                    if args.model == 'EEGPT' and valid_indices is not None:
                        eeg = torch.index_select(eeg, 1, valid_indices)
                    (B, C, A, T) = eeg.shape
                    if args.embedding_high_level:
                        (mask_eeg, mask) = masking_strategy(eeg)
                        if eeg.dim() == 4:
                            eeg = eeg.flatten(2)
                        if mask_eeg.dim() == 4:
                            mask_eeg = mask_eeg.flatten(2)
                        chans_id = encoder_sub.prepare_chan_ids(ch_names)
                        chans_id = chans_id.to(device)
                        val_embedding_sub = encoder_sub(mask_eeg, chans_id.to(mask_eeg))
                        val_embedding_group = encoder_group(eeg, chans_id.to(eeg))
                        (val_loss_curr, val_loss_intra_curr, val_loss_cross_curr) = contrastive_4d_channel_dynamics(val_embedding_sub, val_embedding_group, label, args, epoch=epoch, warmup_epoch=args.warmup_epochs, temperature=args.contrastive_task_temperature, alpha=args.contrastive_task_alpha, batch_size=B, channel=C, patch=A, dimension=T)
                    else:
                        (val_loss_curr, val_loss_intra_curr, val_loss_cross_curr) = contrastive_dynamics(val_embedding_sub, val_embedding_group, label, epoch=epoch, warmup_epoch=args.warmup_epochs, temperature=args.contrastive_task_temperature, alpha=args.contrastive_task_alpha)
                    val_loss += val_loss_curr.item()
                    val_loss_intra += val_loss_intra_curr.item()
                    val_loss_cross += val_loss_cross_curr.item()
            avg_val_loss = val_loss / len(val_loader)
            avg_val_loss_intra = val_loss_intra / len(val_loader)
            avg_val_loss_cross = val_loss_cross / len(val_loader)
            val_losses.append(avg_val_loss)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f'Epoch {epoch + 1}/{args.epochs}: Train Loss={avg_train_loss:.4f}, Loss MSE={avg_train_loss_intra:.4f}, Loss Cross={avg_train_loss_cross:.4f}, Val Loss={avg_val_loss:.4f}, LR={current_lr:.6f}')
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save({'epoch': epoch, 'encoder_state_dict': encoder_sub.state_dict(), 'val_loss': avg_val_loss}, os.path.join(subject_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f'Early stopping at epoch {epoch + 1}')
                    break
        losses = {'train_losses': train_losses, 'val_losses': val_losses}
        if test_metrics is not None:
            print(f"\n{'=' * 80}")
            print(f'Subject {subject_id:02d} - Final Test Evaluation')
            print(f"{'=' * 80}")
            checkpoint = torch.load(os.path.join(subject_dir, 'best_model.pth'))
            encoder_sub.load_state_dict(checkpoint['encoder_state_dict'])
            test_loss = 0
            test_loss_intra = 0
            test_loss_cross = 0
            encoder_sub.eval()
            encoder_group.eval()
            with torch.no_grad():
                for batch in test_loader:
                    eeg = batch['eeg'].to(device)
                    label = batch['label'].to(device)
                    if args.model == 'EEGPT' and valid_indices is not None:
                        eeg = torch.index_select(eeg, 1, valid_indices)
                    (B, C, A, T) = eeg.shape
                    if args.embedding_high_level:
                        (mask_eeg, mask) = masking_strategy(eeg)
                        if eeg.dim() == 4:
                            eeg = eeg.flatten(2)
                        if mask_eeg.dim() == 4:
                            mask_eeg = mask_eeg.flatten(2)
                        chans_id = encoder_sub.prepare_chan_ids(ch_names)
                        chans_id = chans_id.to(device)
                        test_embedding_sub = encoder_sub(mask_eeg, chans_id.to(mask_eeg))
                        test_embedding_group = encoder_group(eeg, chans_id.to(eeg))
                        (test_loss_curr, test_loss_intra_curr, test_loss_cross_curr) = contrastive_4d_channel_dynamics(test_embedding_sub, test_embedding_group, label, args, epoch=epoch, warmup_epoch=args.warmup_epochs, temperature=args.contrastive_task_temperature, alpha=args.contrastive_task_alpha, batch_size=B, channel=C, patch=A, dimension=T)
                    else:
                        (test_loss_curr, test_loss_intra_curr, test_loss_cross_curr) = contrastive_dynamics(test_embedding_sub, test_embedding_group, label, epoch=epoch, warmup_epoch=args.warmup_epochs, temperature=args.contrastive_task_temperature, alpha=args.contrastive_task_alpha)
                    test_loss += test_loss_curr.item()
                    test_loss_intra += test_loss_intra_curr.item()
                    test_loss_cross += test_loss_cross_curr.item()
            avg_test_loss = test_loss / len(test_loader)
            avg_test_loss_intra = test_loss_intra / len(test_loader)
            avg_test_loss_cross = test_loss_cross / len(test_loader)
            metrics = {'subject_id': int(subject_id), 'train_size': train_size, 'train_loss': avg_train_loss, 'train_loss_intra': avg_train_loss_intra, 'train_loss_cross': avg_train_loss_cross, 'val_size': val_size, 'val_loss': avg_val_loss, 'val_loss_intra': avg_val_loss_intra, 'val_loss_cross': avg_val_loss_cross, 'test_size': test_size, 'test_loss': avg_test_loss, 'test_loss_intra': avg_test_loss_intra, 'test_loss_cross': avg_test_loss_cross, 'best_epoch': checkpoint['epoch'] + 1}
            with open(os.path.join(subject_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f'  ✓ Results saved to {subject_dir}')
        else:
            metrics = {'subject_id': int(subject_id), 'train_size': train_size, 'train_loss': avg_train_loss, 'val_size': val_size, 'val_loss': avg_val_loss, 'best_epoch': checkpoint['epoch'] + 1}
            with open(os.path.join(subject_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f'  ✓ Results saved to {subject_dir}')
        torch.cuda.empty_cache()
        return (metrics, losses)
    except Exception as e:
        print(f'  ✗ ERROR for Subject {subject_id:02d}: {str(e)}')
        import traceback
        traceback.print_exc()
        if 'encoder_sub' in locals():
            del encoder_sub
        torch.cuda.empty_cache()
        return (None, None)


def train_single_subject_baseline_multitask(args, subject_id, device, num_patches, dataset):
    """Train a subject classifier with the selected task adaptation method"""
    if args.dataset == 'm3cv':
        label_mapping = {10: 0, 11: 1, 12: 2, 20: 3, 21: 4, 30: 5, 31: 6, 32: 7, 40: 8, 41: 9, 42: 10}
        target_task_ids = [1, 2, 3, 4]
    else:
        label_mapping = {10: 0, 11: 1, 12: 2, 20: 3, 21: 4, 22: 5}
        target_task_ids = [1, 2]
    try:
        output_path = os.path.join(args.output_root, args.dataset, args.cls_fitune_type)
        subject_dir = os.path.join(output_path, f'subject_{subject_id:02d}')
        os.makedirs(subject_dir, exist_ok=True)
        metrics_path = os.path.join(subject_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            print(f'  ✓ Subject {subject_id:02d} already trained, skipping...')
            with open(metrics_path, 'r') as f:
                return json.load(f)
        print(f"\n{'=' * 80}")
        print(f'Subject {subject_id:02d} - Loading Data')
        print(f"{'=' * 80}")
        if args.multi_task:
            (train_loader, val_loader, test_loader) = create_multitask_single_subject_dataloaders(dataset=dataset, target_subject_id=subject_id, target_task_id=target_task_ids, train_size=args.train_size, test_size=args.test_size, val_size=args.val_size, batch_size=args.batch_size, random_state=args.seed, num_workers=args.num_workers)
        elif args.fewshot:
            (train_loader, val_loader, test_loader) = create_single_subject_few_shot_dataloaders(dataset=dataset, target_subject_id=subject_id, n_shot=args.n_shot, batch_size=args.batch_size, random_state=args.seed, num_workers=args.num_workers, val_size=args.val_size, oversample=args.oversample)
        elif args.multitask_fewshot:
            (train_loader, val_loader, test_loader) = create_multitask_single_subject_few_shot_dataloaders(dataset=dataset, target_subject_id=subject_id, target_task_id=target_task_ids, n_shot=args.n_shot, batch_size=args.batch_size, random_state=args.seed, num_workers=args.num_workers, val_size=args.val_size, oversample=args.oversample)
        elif args.dataset_suffle:
            (train_loader, val_loader, test_loader) = create_single_subject_dataloaders(dataset=dataset, target_subject_id=subject_id, train_size=args.train_size, test_size=args.test_size, val_size=args.val_size, batch_size=args.batch_size, random_state=args.seed, num_workers=args.num_workers)
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)
        print(f'  Train: {train_size}, Val: {val_size}, Test: {test_size}')
        if train_size < args.min_samples_per_class:
            raise ValueError(f'Insufficient training samples: {train_size}')
        if args.dataset == 'cogbci':
            (in_chans, ch_names, chan_number) = get_cogbci_dataset(args, subject_id)
            print('in_chans', in_chans)
        else:
            (in_chans, ch_names, chan_number) = get_dataset_name(args)
            print('in_chans', in_chans)
        valid_indices = None
        from src.eegpt import CHANNEL_DICT
        valid_indices_list = []
        valid_ch_names = []
        for (i, name) in enumerate(ch_names):
            clean_name = name.upper().strip('.')
            if clean_name in CHANNEL_DICT:
                valid_indices_list.append(i)
                valid_ch_names.append(name)
            else:
                print(f"  Skipping channel '{name}': not in the EEGPT channel dictionary.")
        if len(valid_indices_list) == 0:
            raise ValueError('No valid EEGPT channels remain; check the channel names.')
        ch_names = valid_ch_names
        valid_indices = torch.tensor(valid_indices_list).long().to(device)
        print(f'  Retained channels: {len(ch_names)}')
        from functools import partial
        from src.eegpt import EEGTransformer
        current_num_channels = len(ch_names)
        encoder = EEGTransformer(img_size=[current_num_channels, 200], patch_size=64, patch_stride=64, embed_num=4, embed_dim=512, depth=8, num_heads=8, mlp_ratio=4.0, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, init_std=0.02, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-06)).to(device)
        pretrain_ckpt = torch.load(args.pretrained_path, weights_only=False)
        target_encoder_stat = {}
        for (k, v) in pretrain_ckpt['state_dict'].items():
            if k.startswith('target_encoder.'):
                target_encoder_stat[k[15:]] = v
        encoder.load_state_dict(target_encoder_stat)
        if args.cls_fitune_type == 'lora':
            encoder = create_lora(encoder, args)
            for (name, param) in encoder.named_parameters():
                if 'lora' not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        elif args.cls_fitune_type == 'full':
            for (name, param) in encoder.named_parameters():
                param.requires_grad = True
        elif args.cls_fitune_type == 'linear':
            for (name, param) in encoder.named_parameters():
                param.requires_grad = False
        model = SimpleClassifier_EEGPT(encoder=encoder, in_chans=in_chans, args=args, use_channels_names=ch_names, num_classes=args.num_classes).to(device)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(f'\n  Trainable parameters: {sum((p.numel() for p in trainable_params)):,}')
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        from torch.cuda.amp import GradScaler
        scaler = GradScaler() if args.use_amp else None
        best_val_acc = float("-inf")
        best_val_f1 = float("-inf")
        patience_counter = 0
        print(f"\n{'=' * 80}")
        print(f'Subject {subject_id:02d} - Training task adaptation')
        print(f'Method: {args.cls_fitune_type.upper()}')
        print(f"{'=' * 80}")
        for epoch in range(args.epochs):
            if epoch < args.warmup_epochs:
                lr_mult = (epoch + 1) / args.warmup_epochs
                current_lr = args.lr * lr_mult
            else:
                progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
                current_lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            model.train()
            train_loss = 0
            for batch in train_loader:
                if isinstance(batch, (tuple, list)):
                    (eeg, labels) = batch
                else:
                    eeg = batch['eeg']
                    labels = batch['label']
                    labels = torch.tensor([label_mapping[label.item()] for label in labels])
                eeg = eeg.to(device)
                labels = labels.to(device)
                if valid_indices is not None:
                    eeg = torch.index_select(eeg, 1, valid_indices)
                if eeg.dim() == 4:
                    eeg = eeg.flatten(2)
                optimizer.zero_grad()
                if args.use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = model(eeg)
                        loss = criterion(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(eeg)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)
            if args.dataset == 'm3cv':
                (val_loss, val_acc, val_f1, val_cm) = evaluate_multitask(model, val_loader, criterion, device, args.use_amp, args.model, valid_indices)
            else:
                (val_loss, val_acc, val_f1, val_cm) = evaluate_multitask_cog(model, val_loader, criterion, device, args.use_amp, args.model, valid_indices)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f'  Epoch {epoch + 1:3d}/{args.epochs}: TrainLoss={avg_train_loss:.4f}, ValAcc={val_acc:.4f}, ValF1={val_f1:.4f}, LR={current_lr:.6f}')
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                patience_counter = 0
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'classifier_state_dict': model.classifier.state_dict(), 'val_acc': val_acc, 'val_f1': val_f1}, os.path.join(subject_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f'  Early stopping at epoch {epoch + 1}')
                    break
        print(f"\n{'=' * 80}")
        print(f'Subject {subject_id:02d} - Final Test Evaluation')
        print(f"{'=' * 80}")
        checkpoint = torch.load(os.path.join(subject_dir, 'best_model.pth'), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        if args.dataset == 'm3cv':
            (test_loss, test_acc, test_f1, test_cm) = evaluate_multitask(model, test_loader, criterion, device, args.use_amp, args.model, valid_indices)
        else:
            (test_loss, test_acc, test_f1, test_cm) = evaluate_multitask_cog(model, test_loader, criterion, device, args.use_amp, args.model, valid_indices)
        print(f'  Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}')
        print(f'  Confusion Matrix:\n{test_cm}')
        metrics = {'subject_id': int(subject_id), 'method': args.cls_fitune_type, 'train_size': train_size, 'val_size': val_size, 'test_size': test_size, 'best_val_acc': float(best_val_acc), 'best_val_f1': float(best_val_f1), 'test_acc': float(test_acc), 'test_f1': float(test_f1), 'test_cm': test_cm.tolist(), 'best_epoch': int(checkpoint['epoch']) + 1, 'transfer_method': 'linear_probing', 'source_task': 'resting_reconstruction', 'target_task': 'cognitive_classification'}
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f'  ✓ Results saved to {subject_dir}')
        del model
        del optimizer
        del encoder
        torch.cuda.empty_cache()
        return metrics
    except Exception as e:
        print(f'  ✗ ERROR for Subject {subject_id:02d}: {str(e)}')
        import traceback
        traceback.print_exc()
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()
        return None


def train_single_subject_cls_multi_task_cog(args, subject_id, device, num_patches, dataset):
    """Train a subject classifier with the selected task adaptation method"""
    label_mapping = {10: 0, 11: 1, 12: 2, 20: 3, 21: 4, 22: 5}
    try:
        valid_fitune_types = ['lora', 'full']
        valid_cls_fitune_types = ['lora', 'full', 'linear']
        if args.fitune_type not in valid_fitune_types:
            raise ValueError(f'fitune_type must be one of {valid_fitune_types}')
        if args.cls_fitune_type not in valid_cls_fitune_types:
            raise ValueError(f'cls_fitune_type must be one of {valid_cls_fitune_types}')
        output_path = os.path.join(args.output_root, args.dataset, args.cls_fitune_type)
        subject_dir = os.path.join(output_path, f'subject_{subject_id:02d}')
        os.makedirs(subject_dir, exist_ok=True)
        metrics_path = os.path.join(subject_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            print(f'  ✓ Subject {subject_id:02d} already trained, skipping...')
            with open(metrics_path, 'r') as f:
                return json.load(f)
        print(f"\n{'=' * 80}")
        print(f'Subject {subject_id:02d} - Loading Data')
        print(f"{'=' * 80}")
        if args.multi_task:
            (train_loader, val_loader, test_loader) = create_multitask_single_subject_dataloaders(dataset=dataset, target_subject_id=subject_id, target_task_id=[1, 2], train_size=args.train_size, test_size=args.test_size, val_size=args.val_size, batch_size=args.batch_size, random_state=args.seed, num_workers=args.num_workers)
        elif args.multitask_fewshot:
            (train_loader, val_loader, test_loader) = create_multitask_single_subject_few_shot_dataloaders(dataset=dataset, target_subject_id=subject_id, target_task_id=[1, 2], n_shot=args.n_shot, batch_size=args.batch_size, random_state=args.seed, num_workers=args.num_workers, val_size=args.val_size, oversample=args.oversample)
        elif args.fewshot:
            (train_loader, val_loader, test_loader) = create_single_subject_few_shot_dataloaders(dataset=dataset, target_subject_id=subject_id, n_shot=args.n_shot, batch_size=args.batch_size, random_state=args.seed, num_workers=args.num_workers, val_size=args.val_size, oversample=args.oversample)
        elif args.dataset_suffle:
            (train_loader, val_loader, test_loader) = create_single_subject_dataloaders(dataset=dataset, target_subject_id=subject_id, train_size=args.train_size, test_size=args.test_size, val_size=args.val_size, batch_size=args.batch_size, random_state=args.seed, num_workers=args.num_workers)
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)
        print(f'  Train: {train_size}, Val: {val_size}, Test: {test_size}')
        if train_size < args.min_samples_per_class:
            raise ValueError(f'Insufficient training samples: {train_size}')
        if args.dataset == 'cogbci':
            (in_chans, ch_names, chan_number) = get_cogbci_dataset(args, subject_id)
            print('in_chans', in_chans)
        else:
            (in_chans, ch_names, chan_number) = get_dataset_name(args)
            print('in_chans', in_chans)
        valid_indices = None
        from src.eegpt import CHANNEL_DICT
        valid_indices_list = []
        valid_ch_names = []
        for (i, name) in enumerate(ch_names):
            clean_name = name.upper().strip('.')
            if clean_name in CHANNEL_DICT:
                valid_indices_list.append(i)
                valid_ch_names.append(name)
            else:
                print(f"  Skipping channel '{name}': not in the EEGPT channel dictionary.")
        if len(valid_indices_list) == 0:
            raise ValueError('No valid EEGPT channels remain; check the channel names.')
        ch_names = valid_ch_names
        valid_indices = torch.tensor(valid_indices_list).long().to(device)
        print(f'  Retained channels: {len(ch_names)}')
        checkpoint_path = os.path.join(args.recon_results_dir, args.dataset, f'subject_{subject_id:02d}', 'best_model.pth')
        print(f'  Checkpoint: {checkpoint_path}')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f'Pretrained encoder checkpoint not found: {checkpoint_path}')
        print(f'\n[Step 1] Creating Encoder ({args.model})...')
        from functools import partial
        current_num_channels = len(ch_names)
        print('current_num_channels', current_num_channels)
        encoder = EEGTransformer(img_size=[current_num_channels, 200], patch_size=64, patch_stride=64, embed_num=4, embed_dim=512, depth=8, num_heads=8, mlp_ratio=4.0, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, init_std=0.02, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-06)).to(device)
        if args.fitune_type == 'lora':
            encoder = create_lora(encoder, args)
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            if args.cls_fitune_type == 'lora':
                for (name, param) in encoder.named_parameters():
                    if 'lora' not in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            elif args.cls_fitune_type == 'full':
                for (name, param) in encoder.named_parameters():
                    if param.is_floating_point():
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            elif args.cls_fitune_type == 'linear':
                for (name, param) in encoder.named_parameters():
                    param.requires_grad = False
        elif args.fitune_type == 'full':
            encoder = load_pretrained_encoder(encoder, checkpoint_path, args.fitune_type, device)
            if args.cls_fitune_type == 'lora':
                encoder = create_lora_cls(encoder, args)
                for (name, param) in encoder.named_parameters():
                    if 'lora' not in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            elif args.cls_fitune_type == 'full':
                for (name, param) in encoder.named_parameters():
                    if param.is_floating_point():
                        param.requires_grad = True
            elif args.cls_fitune_type == 'linear':
                for (name, param) in encoder.named_parameters():
                    param.requires_grad = False
        model = SimpleClassifier_EEGPT(encoder=encoder, in_chans=in_chans, args=args, use_channels_names=ch_names, num_classes=args.num_classes).to(device)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        from torch.cuda.amp import GradScaler
        scaler = GradScaler() if args.use_amp else None
        best_val_acc = float("-inf")
        best_val_f1 = float("-inf")
        patience_counter = 0
        print(f"\n{'=' * 80}")
        print(f'Subject {subject_id:02d} - Training task adaptation')
        print(f'Method: {args.cls_fitune_type.upper()}')
        print(f"{'=' * 80}")
        for epoch in range(args.epochs):
            if epoch < args.warmup_epochs:
                lr_mult = (epoch + 1) / args.warmup_epochs
                current_lr = args.lr * lr_mult
            else:
                progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
                current_lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            model.train()
            train_loss = 0
            for batch in train_loader:
                if isinstance(batch, (tuple, list)):
                    (eeg, labels) = batch
                else:
                    eeg = batch['eeg']
                    labels = batch['label']
                    mapped_labels = torch.tensor([label_mapping[label.item()] for label in labels])
                eeg = eeg.to(device)
                mapped_labels = mapped_labels.to(device)
                if valid_indices is not None:
                    eeg = torch.index_select(eeg, 1, valid_indices)
                if eeg.dim() == 4:
                    eeg = eeg.flatten(2)
                optimizer.zero_grad()
                if args.use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = model(eeg)
                        loss = criterion(logits, mapped_labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(eeg)
                    loss = criterion(logits, mapped_labels)
                    loss.backward()
                    optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)
            (val_loss, val_acc, val_f1, val_cm) = evaluate_multitask_cog(model, val_loader, criterion, device, args.use_amp, model_name=args.model, valid_indices=valid_indices)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f'  Epoch {epoch + 1:3d}/{args.epochs}: TrainLoss={avg_train_loss:.4f}, ValAcc={val_acc:.4f}, ValF1={val_f1:.4f}, LR={current_lr:.6f}')
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                patience_counter = 0
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_acc': val_acc, 'val_f1': val_f1}, os.path.join(subject_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f'  Early stopping at epoch {epoch + 1}')
                    break
        print(f"\n{'=' * 80}")
        print(f'Subject {subject_id:02d} - Final Test Evaluation')
        print(f"{'=' * 80}")
        checkpoint = torch.load(os.path.join(subject_dir, 'best_model.pth'), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        (test_loss, test_acc, test_f1, test_cm) = evaluate_multitask_cog(model, test_loader, criterion, device, args.use_amp, model_name=args.model, valid_indices=valid_indices)
        print(f'  Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}')
        print(f'  Confusion Matrix:\n{test_cm}')
        metrics = {'subject_id': int(subject_id), 'method': args.cls_fitune_type, 'train_size': train_size, 'val_size': val_size, 'test_size': test_size, 'best_val_acc': float(best_val_acc), 'best_val_f1': float(best_val_f1), 'test_acc': float(test_acc), 'test_f1': float(test_f1), 'test_cm': test_cm.tolist(), 'best_epoch': int(checkpoint['epoch']) + 1, 'transfer_method': 'linear_probing', 'source_task': 'resting_reconstruction', 'target_task': 'cognitive_classification'}
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f'  ✓ Results saved to {subject_dir}')
        del model
        del optimizer
        del encoder
        torch.cuda.empty_cache()
        return metrics
    except Exception as e:
        print(f'  ✗ ERROR for Subject {subject_id:02d}: {str(e)}')
        import traceback
        traceback.print_exc()
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()
        return None
