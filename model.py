#!/usr/bin/env python3
"""
Multi-label MS-TCN++ baseline.

Design goal:
- Keep the official MS-TCN++ architecture as much as possible.
- Keep the DDL Prediction_Generation stage unchanged in spirit.
- Keep Refinement stages as prediction-sequence refiners: previous stage prediction -> next stage.
- Replace only the parts that are inherently single-label:
    softmax -> sigmoid
    CrossEntropyLoss -> BCEWithLogitsLoss
    argmax/accuracy -> sigmoid scores + AP/mAP

Expected tensor shapes for multi-label training:
    batch_input:  [B, D, T]
    batch_target: [B, C, T] multi-hot 0/1 labels
    mask:         [B, 1, T] or [B, C, T], where valid positions are 1

The model outputs raw logits, not probabilities:
    predictions: [S, B, C, T]
where S = 1 + num_R stages.
"""

import sys
import copy
import os
from typing import Optional, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from loguru import logger
from sklearn.metrics import average_precision_score


class MS_TCN2(nn.Module):
    """Official MS-TCN++ structure, adapted for multi-label output.

    Important change from official single-label version:
        Refinement input uses sigmoid(out) instead of softmax(out).

    Important thing kept from the official version:
        Refinement stages only consume the previous stage prediction sequence.
        They do NOT receive the original feature again.
    """

    def __init__(
        self,
        num_layers_PG: int,
        num_layers_R: int,
        num_R: int,
        num_f_maps: int,
        dim: int,
        num_classes: int,
        refinement_input: str = "sigmoid",
    ):
        super(MS_TCN2, self).__init__()
        self.PG = Prediction_Generation(num_layers_PG, num_f_maps, dim, num_classes)
        self.Rs = nn.ModuleList(
            [
                copy.deepcopy(
                    Refinement(num_layers_R, num_f_maps, num_classes, num_classes)
                )
                for _ in range(num_R)
            ]
        )
        if refinement_input not in {"sigmoid", "logits"}:
            raise ValueError("refinement_input must be 'sigmoid' or 'logits'")
        self.refinement_input = refinement_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # out is raw logits: [B, C, T]
        out = self.PG(x)
        outputs = out.unsqueeze(0)

        for R in self.Rs:
            if self.refinement_input == "sigmoid":
                # Closest multi-label analogue of official R(softmax(out)).
                # Each class is independent, so sigmoid replaces softmax.
                refine_in = torch.sigmoid(out)
            else:
                # Optional ablation: preserve logit confidence information.
                refine_in = out

            out = R(refine_in)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)

        return outputs


class Prediction_Generation(nn.Module):
    """MS-TCN++ Prediction Generation stage with Dual Dilated Layers.

    This is intentionally kept closest to the official MS-TCN++ implementation:
    every layer has two dilated temporal convolution branches:
        branch 1: large-to-small dilation
        branch 2: small-to-large dilation
    Their outputs are concatenated and fused by a 1x1 convolution.
    """

    def __init__(self, num_layers: int, num_f_maps: int, dim: int, num_classes: int):
        super(Prediction_Generation, self).__init__()
        self.num_layers = num_layers
        self.conv_1x1_in = nn.Conv1d(dim, num_f_maps, 1)

        self.conv_dilated_1 = nn.ModuleList(
            [
                nn.Conv1d(
                    num_f_maps,
                    num_f_maps,
                    3,
                    padding=2 ** (num_layers - 1 - i),
                    dilation=2 ** (num_layers - 1 - i),
                )
                for i in range(num_layers)
            ]
        )

        self.conv_dilated_2 = nn.ModuleList(
            [
                nn.Conv1d(
                    num_f_maps,
                    num_f_maps,
                    3,
                    padding=2 ** i,
                    dilation=2 ** i,
                )
                for i in range(num_layers)
            ]
        )

        self.conv_fusion = nn.ModuleList(
            [nn.Conv1d(2 * num_f_maps, num_f_maps, 1) for _ in range(num_layers)]
        )

        self.dropout = nn.Dropout()
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.conv_1x1_in(x)

        for i in range(self.num_layers):
            f_in = f
            f = self.conv_fusion[i](
                torch.cat([self.conv_dilated_1[i](f), self.conv_dilated_2[i](f)], dim=1)
            )
            f = F.relu(f)
            f = self.dropout(f)
            f = f + f_in

        out = self.conv_out(f)
        return out


class Refinement(nn.Module):
    """Official MS-TCN++ refinement stage.

    Kept as prediction-sequence refinement. It takes the previous stage prediction
    representation as input, not the original video feature.
    """

    def __init__(self, num_layers: int, num_f_maps: int, dim: int, num_classes: int):
        super(Refinement, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [
                copy.deepcopy(
                    DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)
                )
                for i in range(num_layers)
            ]
        )
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation: int, in_channels: int, out_channels: int):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(
            in_channels, out_channels, 3, padding=dilation, dilation=dilation
        )
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


# -----------------------------------------------------------------------------
# Optional legacy MS_TCN classes.
# Trainer below uses MS_TCN2, not MS_TCN. These are kept only for compatibility
# with the official file structure.
# -----------------------------------------------------------------------------
class MS_TCN(nn.Module):
    def __init__(self, num_stages: int, num_layers: int, num_f_maps: int, dim: int, num_classes: int):
        super(MS_TCN, self).__init__()
        self.stage1 = SS_TCN(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList(
            [
                copy.deepcopy(SS_TCN(num_layers, num_f_maps, num_classes, num_classes))
                for _ in range(num_stages - 1)
            ]
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for stage in self.stages:
            refine_in = torch.sigmoid(out)
            if mask is not None:
                refine_in = refine_in * mask[:, 0:1, :]
            out = stage(refine_in, mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SS_TCN(nn.Module):
    def __init__(self, num_layers: int, num_f_maps: int, dim: int, num_classes: int):
        super(SS_TCN, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [
                copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps))
                for i in range(num_layers)
            ]
        )
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        if mask is not None:
            out = out * mask[:, 0:1, :]
        return out


class Trainer:
    """Trainer for multi-label frame-wise action segmentation.

    This Trainer expects multi-hot targets instead of single class indices.
    It reports AP/mAP instead of single-label frame accuracy.

    It supports two constructor styles:

    Official-style main_origin.py:
        Trainer(num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes, dataset, split)

    Your modified main.py style:
        Trainer(args, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes, dataset, split, device=device)

    Optional args attributes supported when using the second style:
        args.bce_pos_weight
        args.smoothing_weight
        args.smoothing_tau
        args.refinement_input
    """

    def __init__(self, *init_args, **kwargs):
        # ------------------------------------------------------------------
        # Parse either official-style or args-first-style constructor.
        # ------------------------------------------------------------------
        if len(init_args) >= 9 and not isinstance(init_args[0], int):
            cfg = init_args[0]
            (
                num_layers_PG,
                num_layers_R,
                num_R,
                num_f_maps,
                dim,
                num_classes,
                dataset,
                split,
            ) = init_args[1:9]

            pos_weight = kwargs.get("pos_weight", None)
            bce_pos_weight = kwargs.get(
                "bce_pos_weight", getattr(cfg, "bce_pos_weight", 1.0)
            )
            smoothing_weight = kwargs.get(
                "smoothing_weight", getattr(cfg, "smoothing_weight", 0.15)
            )
            smoothing_tau = kwargs.get(
                "smoothing_tau", getattr(cfg, "smoothing_tau", 4.0)
            )
            refinement_input = kwargs.get(
                "refinement_input", getattr(cfg, "refinement_input", "sigmoid")
            )
            save_best_only = kwargs.get(
                "save_best_only", getattr(cfg, "save_best_only", False)
            )
            save_every = kwargs.get(
                "save_every", getattr(cfg, "save_every", 1)
            )
            save_optimizer = kwargs.get(
                "save_optimizer", getattr(cfg, "save_optimizer", False)
            )
            batch_size = kwargs.get(
                "bz", getattr(cfg, "bz", False)
            )
            learning_rate = kwargs.get(
                "lr", getattr(cfg, "lr", False)
            )
            num_epochs = kwargs.get(
                "num_epochs", getattr(cfg, "num_epochs", False)
            )
            stage_diagnostics = kwargs.get(
                "stage_diagnostics", getattr(cfg, "stage_diagnostics", False)
            )

        else:
            if len(init_args) < 8:
                raise TypeError(
                    "Trainer expects either "
                    "Trainer(num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes, dataset, split) "
                    "or Trainer(args, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes, dataset, split)."
                )
            (
                num_layers_PG,
                num_layers_R,
                num_R,
                num_f_maps,
                dim,
                num_classes,
                dataset,
                split,
            ) = init_args[:8]

            pos_weight = kwargs.get("pos_weight", None)
            bce_pos_weight = kwargs.get("bce_pos_weight", 1.0)
            smoothing_weight = kwargs.get("smoothing_weight", 0.15)
            smoothing_tau = kwargs.get("smoothing_tau", 4.0)
            refinement_input = kwargs.get("refinement_input", "sigmoid")
            save_best_only = kwargs.get("save_best_only", False)
            save_every = kwargs.get("save_every", 1)
            save_optimizer = kwargs.get("save_optimizer", False)
            stage_diagnostics = kwargs.get("stage_diagnostics", False)

        self.model = MS_TCN2(
            num_layers_PG,
            num_layers_R,
            num_R,
            num_f_maps,
            dim,
            num_classes,
            refinement_input=refinement_input,
        )
        self.num_classes = num_classes
        self.mse = nn.MSELoss(reduction="none")
        self.smoothing_weight = smoothing_weight
        self.smoothing_tau = smoothing_tau

        if num_R == 0:
            refinement_tag = "PGonly"
        else:
            refinement_tag = refinement_input

        self.checkpoint_tag = (
            f"R{num_R}"
            f"_{refinement_tag}"
            f"_pw{float(bce_pos_weight):g}"
            f"_s{float(smoothing_weight):g}"
            f"_tau{float(smoothing_tau):g}"
            f"_lr{float(learning_rate):g}"
            f"_bz{int(batch_size)}"
        )

        # Checkpoint policy.
        # save_best_only=True: only save best.model based on validation mAP.
        # save_every=N: when save_best_only=False, save epoch checkpoints every N epochs.
        # save_optimizer=True: also save optimizer state for resumable training.
        self.save_best_only = bool(save_best_only)
        self.save_every = int(save_every) if save_every is not None else 0
        self.save_optimizer = bool(save_optimizer)
        self.stage_diagnostics = bool(stage_diagnostics)
        if self.save_every < 0:
            raise ValueError("save_every must be >= 0. Use 0 to disable periodic epoch checkpoints.")

        if pos_weight is None:
            self.pos_weight = torch.ones(num_classes, dtype=torch.float32) * float(bce_pos_weight)
        else:
            self.pos_weight = torch.as_tensor(pos_weight, dtype=torch.float32)
            if self.pos_weight.numel() != num_classes:
                raise ValueError(
                    f"pos_weight must have {num_classes} values, got {self.pos_weight.numel()}"
                )

        os.makedirs("logs", exist_ok=True)

        
        logger.add("logs/" + str(dataset) + "_" + str(split) + "_{time}.log")
        logger.add(sys.stdout, colorize=True, format="{message}")

        logger.info("=" * 80)
        logger.info("Experiment Configuration")
        logger.info("=" * 80)

        logger.info(f"dataset = {dataset}")
        logger.info(f"split = {split}")

        logger.info(f"batch_size = {batch_size}")
        logger.info(f"learning_rate = {learning_rate}")
        logger.info(f"num_epochs = {num_epochs}")

        logger.info(f"num_classes = {num_classes}")
        logger.info(f"features_dim = {dim}")
        logger.info(f"num_f_maps = {num_f_maps}")

        logger.info(f"num_layers_PG = {num_layers_PG}")
        logger.info(f"num_layers_R = {num_layers_R}")
        logger.info(f"num_R = {num_R}")

        logger.info(f"refinement_input = {refinement_input}")

        logger.info(f"bce_pos_weight = {bce_pos_weight}")
        logger.info(f"smoothing_weight = {smoothing_weight}")
        logger.info(f"smoothing_tau = {smoothing_tau}")
        logger.info(f"stage_diagnostics = {self.stage_diagnostics}")


    def _prepare_target_and_mask(
        self,
        batch_target: torch.Tensor,
        mask: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert target/mask into [B, C, T] float tensors.

        Expected target is already [B, C, T]. If your dataloader returns [B, T, C],
        this function will automatically permute it.
        """
        batch_target = batch_target.to(device)
        mask = mask.to(device)

        if batch_target.dim() != 3:
            raise ValueError(
                "For multi-label training, batch_target must be [B, C, T] or [B, T, C]. "
                f"Got shape {tuple(batch_target.shape)}"
            )

        if batch_target.size(1) == self.num_classes:
            target = batch_target.float()
        elif batch_target.size(2) == self.num_classes:
            target = batch_target.permute(0, 2, 1).contiguous().float()
        else:
            raise ValueError(
                "Cannot infer class dimension in batch_target. "
                f"Expected C={self.num_classes}, got shape {tuple(batch_target.shape)}"
            )

        if mask.dim() == 2:
            # [B, T] -> [B, C, T]
            mask_out = mask.unsqueeze(1).float().expand(-1, self.num_classes, -1)
        elif mask.dim() == 3:
            if mask.size(1) == 1:
                mask_out = mask.float().expand(-1, self.num_classes, -1)
            elif mask.size(1) == self.num_classes:
                mask_out = mask.float()
            elif mask.size(2) == self.num_classes:
                mask_out = mask.permute(0, 2, 1).contiguous().float()
            else:
                raise ValueError(
                    "Cannot infer class dimension in mask. "
                    f"Expected [B,1,T], [B,C,T], or [B,T,C], got {tuple(mask.shape)}"
                )
        else:
            raise ValueError(f"mask must be [B,T], [B,1,T], or [B,C,T], got {tuple(mask.shape)}")

        if target.shape != mask_out.shape:
            raise ValueError(
                f"target and mask shape mismatch after formatting: {target.shape} vs {mask_out.shape}"
            )

        return target, mask_out

    def _bce_loss(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # logits/target/mask are [B, C, T].
        # self.pos_weight is conceptually either:
        #   - scalar expanded to [C], or
        #   - per-class vector [C].
        # For PyTorch broadcasting with [B, C, T], reshape it to [1, C, 1]
        # so the weights are applied along the class dimension, not the time dimension.
        pos_weight = self.pos_weight.to(logits.device).view(1, self.num_classes, 1)
        bce = F.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=pos_weight,
            reduction="none",
        )
        bce = bce * mask
        return bce.sum() / mask.sum().clamp_min(1.0)

    def _temporal_smoothing_loss(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Multi-label analogue of MS-TCN++ truncated smoothing loss.

        Official single-label version smooths log-softmax probabilities across time.
        For multi-label, each class is an independent sigmoid classifier, so we use
        log-sigmoid probabilities instead.
        """
        if logits.size(-1) <= 1:
            return logits.new_tensor(0.0)

        log_probs_current = F.logsigmoid(logits[:, :, 1:])
        log_probs_previous = F.logsigmoid(logits.detach()[:, :, :-1])

        mse = self.mse(log_probs_current, log_probs_previous)
        mse = torch.clamp(mse, min=0.0, max=self.smoothing_tau ** 2)

        smooth_mask = mask[:, :, 1:]
        return (mse * smooth_mask).sum() / smooth_mask.sum().clamp_min(1.0)

    def _stage_loss(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        loss = self._bce_loss(logits, target, mask)
        if self.smoothing_weight > 0:
            loss = loss + self.smoothing_weight * self._temporal_smoothing_loss(logits, mask)
        return loss

    def _collect_scores(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        per_class_true: Dict[int, List[float]],
        per_class_scores: Dict[int, List[float]],
    ) -> None:
        probs = torch.sigmoid(logits).detach().cpu()
        target_cpu = target.detach().cpu()
        mask_cpu = mask.detach().cpu()

        batch_size, num_classes, _ = probs.shape
        for b in range(batch_size):
            valid_len = int(mask_cpu[b, 0, :].sum().item())
            if valid_len <= 0:
                continue
            for c in range(num_classes):
                y_true = target_cpu[b, c, :valid_len].numpy()
                y_score = probs[b, c, :valid_len].numpy()
                per_class_true[c].extend(y_true.tolist())
                per_class_scores[c].extend(y_score.tolist())

    def _compute_map(
        self,
        per_class_true: Dict[int, List[float]],
        per_class_scores: Dict[int, List[float]],
    ) -> Tuple[float, Dict[int, float]]:
        per_class_ap = {}
        for c in range(self.num_classes):
            y_true = np.asarray(per_class_true[c])
            y_score = np.asarray(per_class_scores[c])

            # If a class has no positive sample in this split, AP is undefined.
            # We skip it from mean mAP, but still report it as 0.0 if needed.
            if y_true.size > 0 and np.sum(y_true) > 0:
                per_class_ap[c] = float(average_precision_score(y_true, y_score))

        mAP = float(np.mean(list(per_class_ap.values()))) if per_class_ap else 0.0
        return mAP, per_class_ap

    def train(
        self,
        save_dir: str,
        batch_gen,
        batch_gen_val=None,
        num_epochs: int = 50,
        batch_size: int = 1,
        learning_rate: float = 0.0005,
        device: Union[str, torch.device] = "cuda",
    ) -> None:
        os.makedirs(save_dir, exist_ok=True)
        device = torch.device(device)
        self.model.train()
        self.model.to(device)

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        best_val_mAP = -1.0

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            per_class_true = {c: [] for c in range(self.num_classes)}
            per_class_scores = {c: [] for c in range(self.num_classes)}

            while batch_gen.has_next():
                batch_input, batch_target, mask = batch_gen.next_batch(batch_size)
                batch_input = batch_input.to(device)
                target, mask = self._prepare_target_and_mask(batch_target, mask, device)

                optimizer.zero_grad()
                predictions = self.model(batch_input)  # [S, B, C, T]

                loss = predictions.new_tensor(0.0)
                for p in predictions:
                    loss = loss + self._stage_loss(p, target, mask)

                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item())
                num_batches += 1
                self._collect_scores(predictions[-1], target, mask, per_class_true, per_class_scores)

            batch_gen.reset()
            train_mAP, _ = self._compute_map(per_class_true, per_class_scores)
            mean_epoch_loss = epoch_loss / max(num_batches, 1)
            
            #
            probs = torch.sigmoid(predictions[-1]).detach()
            logger.info(
                "prob mean %.6f, max %.6f, min %.6f"
                % (probs.mean().item(), probs.max().item(), probs.min().item())
            )
            pos_rate = target.sum().item() / mask.sum().item()
            logger.info("GT positive rate %.6f" % pos_rate)
            #

            # Save periodic epoch checkpoints only when requested.
            # This avoids creating one model file per epoch by default when save_best_only=True.
            should_save_epoch = (
                not self.save_best_only
                and self.save_every > 0
                and ((epoch + 1) % self.save_every == 0 or (epoch + 1) == num_epochs)
            )
            if should_save_epoch:
                torch.save(self.model.state_dict(), os.path.join(save_dir, f"epoch-{epoch + 1}_{self.checkpoint_tag}.model"))
                if self.save_optimizer:
                    torch.save(optimizer.state_dict(), os.path.join(save_dir, f"epoch-{epoch + 1}.opt"))

            if batch_gen_val is not None:
                if self.stage_diagnostics:
                    val_mAP, _, stage_results = self.validate(
                        batch_gen_val,
                        device,
                        stage_diagnostics=True,
                    )
                else:
                    val_mAP, _ = self.validate(batch_gen_val, device)
                    stage_results = None

                if val_mAP > best_val_mAP:
                    best_val_mAP = val_mAP
                    checkpoint_name = f"best_{self.checkpoint_tag}.model"
                    checkpoint_path = os.path.join(
                        save_dir,
                        checkpoint_name
                    )
                    torch.save(
                        self.model.state_dict(),
                        checkpoint_path
                    )
                    if self.save_optimizer:
                        torch.save(optimizer.state_dict(), os.path.join(save_dir, "best.opt"))
                    logger.info(
                        f"New best checkpoint saved: "
                        f"{checkpoint_name}"
                    )

                logger.info(
                    "[epoch %d]: epoch loss = %.6f, train mAP = %.6f, val mAP = %.6f"
                    % (epoch + 1, mean_epoch_loss, train_mAP, val_mAP)
                )

                if self.stage_diagnostics and stage_results is not None:
                    stage_message = ", ".join(
                        f"{stage_name} = {stage_results[stage_name]['mAP']:.6f}"
                        for stage_name in stage_results
                    )
                    logger.info(
                        "[epoch %d]: validation stage mAP: %s"
                        % (epoch + 1, stage_message)
                    )
            else:
                # Without a validation set, best.model is not meaningful.
                # If save_best_only=True, keep the latest model under last.model.
                if self.save_best_only:
                    torch.save(self.model.state_dict(), os.path.join(save_dir, "last.model"))
                    if self.save_optimizer:
                        torch.save(optimizer.state_dict(), os.path.join(save_dir, "last.opt"))
                logger.info(
                    "[epoch %d]: epoch loss = %.6f, train mAP = %.6f"
                    % (epoch + 1, mean_epoch_loss, train_mAP)
                )

    def validate(
        self,
        batch_gen,
        device: Union[str, torch.device],
        stage_diagnostics: bool = False,
    ):
        device = torch.device(device)
        self.model.eval()

        # Normal validation metric: final stage only.
        per_class_true = {c: [] for c in range(self.num_classes)}
        per_class_scores = {c: [] for c in range(self.num_classes)}

        # Optional stage-wise validation diagnostics.
        # predictions[0] = PG, predictions[1] = R1, ...
        stage_names = ["PG"] + [
            f"R{i + 1}" for i in range(len(self.model.Rs))
        ]
        stage_per_class_scores = None

        if stage_diagnostics:
            stage_per_class_scores = {
                stage_name: {c: [] for c in range(self.num_classes)}
                for stage_name in stage_names
            }

        with torch.no_grad():
            while batch_gen.has_next():
                batch_input, batch_target, mask = batch_gen.next_batch(1)
                batch_input = batch_input.to(device)
                target, mask = self._prepare_target_and_mask(batch_target, mask, device)
                predictions = self.model(batch_input)

                # Keep the original validation behavior for checkpoint selection.
                self._collect_scores(
                    predictions[-1],
                    target,
                    mask,
                    per_class_true,
                    per_class_scores,
                )

                # Diagnostics only: collect scores from every stage against
                # the exact same validation targets already stored above.
                if stage_diagnostics:
                    mask_cpu = mask.detach().cpu()
                    for stage_idx, stage_name in enumerate(stage_names):
                        stage_probs = torch.sigmoid(
                            predictions[stage_idx]
                        ).detach().cpu()

                        batch_size, num_classes, _ = stage_probs.shape
                        for b in range(batch_size):
                            valid_len = int(mask_cpu[b, 0, :].sum().item())
                            if valid_len <= 0:
                                continue
                            for c in range(num_classes):
                                stage_per_class_scores[stage_name][c].extend(
                                    stage_probs[b, c, :valid_len].numpy().tolist()
                                )

        batch_gen.reset()

        val_mAP, per_class_ap = self._compute_map(
            per_class_true,
            per_class_scores,
        )

        stage_results = None
        if stage_diagnostics:
            stage_results = {}
            for stage_name in stage_names:
                stage_mAP, stage_ap = self._compute_map(
                    per_class_true,
                    stage_per_class_scores[stage_name],
                )
                stage_results[stage_name] = {
                    "mAP": stage_mAP,
                    "per_class_ap": stage_ap,
                }

        self.model.train()

        if stage_diagnostics:
            return val_mAP, per_class_ap, stage_results
        return val_mAP, per_class_ap

    def predict(
        self,
        model_dir: str,
        results_dir: str,
        features_path: str,
        vid_list_file: str,
        epoch: Union[int, str],
        actions_dict: Dict[str, int],
        device: Union[str, torch.device],
        sample_rate: int,
        gt_path: Optional[str] = None,
        mapping_file: Optional[str] = None,
        threshold: float = 0.5,
        use_best_model: bool = True,
        stage_diagnostics: bool = False,
        model_path: Optional[str] = None,
    ) -> Optional[float]:
        """Predict multi-label frame labels and optionally compute test mAP.

        If gt_path is provided, GT files are expected to contain one multi-hot row per
        frame, e.g. "0 1 0 ...". For sample_rate > 1, GT is downsampled in the
        same way as features before AP/mAP calculation.
        """
        os.makedirs(results_dir, exist_ok=True)
        device = torch.device(device)
        self.model.eval()
        self.model.to(device)

        if model_path is None:
            if use_best_model:
                model_path = os.path.join(
                    model_dir,
                    f"best_{self.checkpoint_tag}.model"
                )
            else:
                model_path = os.path.join(
                    model_dir,
                    f"epoch-{epoch}_{self.checkpoint_tag}.model"
                )

        logger.info(f"Loading prediction model: {model_path}")

        self.model.load_state_dict(
            torch.load(model_path, map_location=device)
        )

        with open(vid_list_file, "r") as f:
            list_of_vids = [
                line.strip() for line in f if line.strip()
            ]

        idx_to_action = {idx: action for action, idx in actions_dict.items()}
        if mapping_file is not None:
            idx_to_action = {}
            with open(mapping_file, "r") as f:
                actions = [
                    line.strip() for line in f if line.strip()
                ]
            for line in actions:
                idx, action_name = line.split()[:2]
                idx_to_action[int(idx)] = action_name

        per_class_true = {c: [] for c in range(self.num_classes)}
        per_class_scores = {c: [] for c in range(self.num_classes)}

        # -------------------------------------------------------------------------
        # Optional stage-wise diagnostics.
        #
        # predictions[0] = PG
        # predictions[1] = R1
        # predictions[2] = R2
        # ...
        #
        # This is only used for analysis and does not change the model prediction.
        # -------------------------------------------------------------------------
        stage_names = ["PG"] + [
            f"R{i + 1}" for i in range(len(self.model.Rs))
        ]

        stage_per_class_scores = None

        if stage_diagnostics:
            if gt_path is None:
                raise ValueError(
                    "stage_diagnostics requires gt_path because AP/mAP needs ground truth."
                )

            stage_per_class_scores = {
                stage_name: {c: [] for c in range(self.num_classes)}
                for stage_name in stage_names
            }

            logger.info(
                "Stage diagnostics enabled: %s"
                % ", ".join(stage_names)
            )
        #

        with torch.no_grad():
            for vid in list_of_vids:
                features = np.load(os.path.join(features_path, vid.split(".")[0] + ".npy"))
                features = features[:, ::sample_rate]

                input_x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
                # predictions = self.model(input_x)
                # probs = torch.sigmoid(predictions[-1]).detach().cpu().numpy()[0]  # [C, T]

                predictions = self.model(input_x)  # [S, 1, C, T]
                # Final-stage prediction.
                # Normal prediction behavior still uses only the last stage.
                probs = (
                    torch.sigmoid(predictions[-1])
                    .detach()
                    .cpu()
                    .numpy()[0]
                )  # [C, T]

                # Only compute all-stage probabilities when diagnostics is enabled.
                if stage_diagnostics:
                    stage_probs = (
                        torch.sigmoid(predictions)
                        .detach()
                        .cpu()
                        .numpy()[:, 0]
                    )  # [S, C, T]
                else:
                    stage_probs = None


                # Optional mAP computation.
                if gt_path is not None:
                    gt_file = os.path.join(gt_path, vid.split("/")[-1].split(".")[0] + ".txt")
                    with open(gt_file, "r") as f:
                        gt_content = [
                            line.strip() for line in f if line.strip()
                        ]

                    gt_labels = np.asarray(
                        [list(map(int, line.split())) for line in gt_content],
                        dtype=np.float32,
                    ).T  # [C, T_full]

                    # Select only the classes used by the model.
                    gt_labels = gt_labels[:self.num_classes, :]
                    
                    gt_labels = gt_labels[:, ::sample_rate]

                    # seq_len = min(probs.shape[1], gt_labels.shape[1])
                    # for c in range(self.num_classes):
                    #     per_class_true[c].extend(gt_labels[c, :seq_len].tolist())
                    #     per_class_scores[c].extend(probs[c, :seq_len].tolist())
                    if probs.shape[1] != gt_labels.shape[1]:
                        logger.warning(
                            f"{vid}: Feature/GT length mismatch: "
                            f"Feature={probs.shape[1]}, "
                            f"GT={gt_labels.shape[1]}. "
                            f"Using min length."
                        )
                    seq_len = min(
                        probs.shape[1],
                        gt_labels.shape[1]
                    )

                    # Ground truth only needs to be stored once because
                    # all stages are evaluated against exactly the same GT.
                    for c in range(self.num_classes):
                        per_class_true[c].extend(
                            gt_labels[c, :seq_len].tolist()
                        )

                        # Normal final-stage metric.
                        per_class_scores[c].extend(
                            probs[c, :seq_len].tolist()
                        )


                    # -------------------------------------------------------------
                    # Optional PG / R1 / R2 / R3 score collection.
                    # -------------------------------------------------------------
                    if stage_diagnostics:
                        for stage_idx, stage_name in enumerate(stage_names):
                            for c in range(self.num_classes):
                                stage_per_class_scores[stage_name][c].extend(
                                    stage_probs[
                                        stage_idx,
                                        c,
                                        :seq_len
                                    ].tolist()
                                )

                else:
                    seq_len = probs.shape[1]

                # Write frame-level multi-label recognition.
                binary_pred = (probs[:, :seq_len] >= threshold).astype(np.int32)
                recognition = []
                for t in range(seq_len):
                    labels = [idx_to_action[i] for i in range(self.num_classes) if binary_pred[i, t] == 1]
                    line = " ".join(labels) if labels else "background"
                    # Match official behavior: if features were sampled, repeat predictions.
                    for _ in range(sample_rate):
                        recognition.append(line)

                f_name = vid.split("/")[-1].split(".")[0]
                with open(os.path.join(results_dir, f_name), "w") as f_ptr:
                    f_ptr.write("\n".join(recognition))

        if gt_path is not None:
            mAP, per_class_ap = self._compute_map(
                per_class_true,
                per_class_scores
            )

            results_file = os.path.join(
                results_dir,
                "per_class_metrics.txt"
            )

            with open(results_file, "w") as f:
                f.write("mean mAP:\n")
                f.write(f"{mAP:.6f}\n")
                f.write("Per-class AP:\n")
                f.write("Class\tAP\n")

                for c in range(self.num_classes):
                    action = idx_to_action.get(c, str(c))
                    ap = per_class_ap.get(c, 0.0)
                    f.write(f"{action}\t{ap:.6f}\n")

            # ---------------------------------------------------------
            # Optional stage-wise diagnostics
            # ---------------------------------------------------------
            if stage_diagnostics:
                stage_results = {}

                for stage_name in stage_names:
                    stage_mAP, stage_ap = self._compute_map(
                        per_class_true,
                        stage_per_class_scores[stage_name]
                    )

                    stage_results[stage_name] = {
                        "mAP": stage_mAP,
                        "per_class_ap": stage_ap,
                    }

                diagnostic_file = os.path.join(
                    results_dir,
                    "stage_diagnostics.txt"
                )

                with open(diagnostic_file, "w") as f:
                    f.write("Stage-wise mAP\n")
                    f.write("====================\n")

                    for stage_name in stage_names:
                        f.write(
                            f"{stage_name}\t"
                            f"{stage_results[stage_name]['mAP']:.6f}\n"
                        )

                    f.write("\n")
                    f.write("Per-class AP by stage\n")
                    f.write("=====================\n")

                    f.write("Class")
                    for stage_name in stage_names:
                        f.write(f"\t{stage_name}")
                    f.write("\n")

                    for c in range(self.num_classes):
                        action = idx_to_action.get(c, str(c))
                        f.write(action)

                        for stage_name in stage_names:
                            ap = stage_results[
                                stage_name
                            ]["per_class_ap"].get(c, 0.0)

                            f.write(f"\t{ap:.6f}")

                        f.write("\n")

                logger.info("=" * 80)
                logger.info("Stage-wise diagnostics")

                for stage_name in stage_names:
                    logger.info(
                        "%s mAP = %.6f"
                        % (
                            stage_name,
                            stage_results[stage_name]["mAP"],
                        )
                    )

                logger.info(
                    "Stage diagnostic report saved to: %s"
                    % diagnostic_file
                )
                logger.info("=" * 80)

            # 要在 diagnostics 做完之後才能 return
            return mAP

        return None