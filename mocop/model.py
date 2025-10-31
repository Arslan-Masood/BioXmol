import decimal
from typing import Iterable, List, Optional, Set, Tuple, Union
import gc
import psutil

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import ListConfig, OmegaConf
from torch import nn as nn
from torch.nn import functional as F
import torch.distributed as dist


from layers import GatedGraphConvolution
from metrics import accuracy
from steps import _supervised_metric, _validation_epoch_end, _validation_epoch_end_all_cell_lines
from pretrained_utils import load_pretrained_encoder, PretrainedEncoder

class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self,y_pred, y_true):
        """
        Focal Loss function for binary classification.

        Arguments:
        y_true -- true binary labels (0 or 1), torch.Tensor
        y_pred -- predicted probabilities for the positive class, torch.Tensor

        Returns:
        Focal Loss
        """
        # Compute class weight
        p = torch.sigmoid(y_pred)


        # Compute focal loss for positive and negative examples
        focal_loss_pos = - (1 - p) ** self.gamma * y_true * torch.log(p.clamp(min=1e-8))
        focal_loss_pos_neg = - p ** self.gamma * (1 - y_true) * torch.log((1 - p).clamp(min=1e-8))

        return focal_loss_pos + focal_loss_pos_neg


def log_memory_usage(pl_module, stage="epoch_start"):
    """Log current process memory usage in GB (not system-wide)."""
    # CPU Memory - Only YOUR process (important for shared clusters)
    process = psutil.Process()
    cpu_memory_gb = process.memory_info().rss / 1024 / 1024 / 1024  # Convert to GB
    
    # Log per-process memory
    pl_module.log(f"{stage}/cpu_memory_gb", cpu_memory_gb, 
                    on_step=False, on_epoch=True, sync_dist=True, reduce_fx="sum")
    

class Hidden_block(nn.Module):
    def __init__(self, input_dim, hidden_dim, norm_type=None, use_skip_connection=True):
        super(Hidden_block, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.use_skip_connection = use_skip_connection

        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim)
        else:
            self.norm = None

    def forward(self, x1):
        x2 = self.layer1(x1)
        if self.norm is not None:
            x2 = self.norm(x2)
        if self.use_skip_connection:
            x2 = x2 + x1
        return x2
    

class MultiLayerPerceptron(nn.Module):
    """Standard multi-layer perceptron with non-linearity and potentially dropout.

    Parameters
    ----------
    num_input_features : int
        input dimension
    num_classes : int, optional
        Number of output classes. If not specified (or None), MLP does not have a final layer.
    hidden_layer_dimensions : List[int], optional
        list of hidden layer dimensions. If not provided, class is a linear model
    nonlin : Union[str, nn.Module]
        name of a nonlinearity in torch.nn, or a pytorch Module. default is relu
    p_dropout : float
        dropout probability for dropout layers. default is 0.0
    """

    def __init__(
        self,
        num_input_features: int,
        num_classes: Optional[int] = None,
        hidden_layer_dimensions: Optional[List[int]] = None,
        nonlin: Union[str, nn.Module] = "ReLU",
        p_dropout: float = 0.0,
        norm_type: Optional[str] = None,
        use_hidden_block: bool = False,
        n_hidden_blocks: int = 1,
        hidden_block_dim: int = 128,
        use_skip_connection: bool = True,
    ):
        super(MultiLayerPerceptron, self).__init__()
        if hidden_layer_dimensions is None:
            hidden_layer_dimensions = []
        if isinstance(hidden_layer_dimensions, ListConfig):
            hidden_layer_dimensions = OmegaConf.to_object(hidden_layer_dimensions)
        if isinstance(nonlin, str):
            nonlin = getattr(torch.nn, nonlin)()

        self.encoder_nonlin = nonlin
        self.encoder_dropout = nn.Dropout(p=p_dropout)
        self.n_hidden_blocks = n_hidden_blocks

        hidden_layer_dimensions = [dim for dim in hidden_layer_dimensions if dim != 0]
        layer_inputs = [num_input_features] + hidden_layer_dimensions
        modules = []
        for i in range(len(hidden_layer_dimensions)):
            modules.append(nn.Dropout(p=p_dropout))
            modules.append(nn.Linear(layer_inputs[i], layer_inputs[i + 1]))
            # Add normalization if requested
            if norm_type == "batchnorm":
                modules.append(nn.BatchNorm1d(layer_inputs[i + 1]))
            elif norm_type == "layernorm":
                modules.append(nn.LayerNorm(layer_inputs[i + 1]))
            if i < (len(hidden_layer_dimensions) - 1):
                modules.append(nonlin)

        self.module = nn.Sequential(*modules)
        if num_classes is None:
            self.has_final_layer = False
        else:
            self.has_final_layer = True
            if num_classes > 1:
                self.output_shape = (num_classes,)
            else:
                self.output_shape = ()
            output_size = num_classes
            self.final_nonlin = nonlin
            self.final_dropout = nn.Dropout(p=p_dropout)
            self.final = nn.Linear(layer_inputs[-1], output_size)

        self.use_hidden_block = use_hidden_block
        if use_hidden_block:
            self.hidden_blocks = nn.ModuleList([
                Hidden_block(
                    input_dim=hidden_block_dim,
                    hidden_dim=hidden_block_dim,
                    norm_type=norm_type,
                    use_skip_connection=use_skip_connection
                )
                for _ in range(self.n_hidden_blocks)
            ])

    def embed(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run forward pass up to penultimate layer"""
        outputs = self.module(inputs)
        return outputs

    def forward(self, x_a: torch.Tensor, **kwargs) -> torch.Tensor:
        # Apply MLP
        outputs = self.module(x_a)
        if self.use_hidden_block:
            # Apply non-linearity and dropout to the final output
            outputs = self.encoder_nonlin(outputs)
            outputs = self.encoder_dropout(outputs)
            # Apply hidden blocks
            for i, block in enumerate(self.hidden_blocks):
                outputs = block(outputs)
                # Apply activation and dropout between blocks, but not after the last one
                if i < self.n_hidden_blocks - 1:
                    outputs = self.encoder_nonlin(outputs)
                    outputs = self.encoder_dropout(outputs)

        if self.has_final_layer:
            # Apply non-linearity and dropout to the final output
            outputs = self.final_nonlin(outputs)
            outputs = self.final_dropout(outputs)
            outputs = self.final(outputs)
            outputs = torch.reshape(outputs, outputs.shape[:-1] + self.output_shape)
        return outputs


class DualInputEncoder(pl.LightningModule):
    def __init__(
        self,
        encoder_a: Optional[MultiLayerPerceptron],
        encoder_b: Optional[MultiLayerPerceptron],
        supervised_head_dim=[64, 2],
        non_lin_proj: bool = False,
        dim=128,
        temperature=10,
    ):
        super().__init__()
        self.encoder_a = encoder_a
        self.encoder_b = encoder_b
        self.dim = dim
        self.supervised_head_dim = supervised_head_dim
        self.add_module("supervised_head", None)
        self.add_module("h_a", None)
        self.add_module("h_b", None)
        self.temperature = temperature
        self.validation_step_outputs = []

        if non_lin_proj:
            self.proj_func = F.relu
        else:
            self.proj_func = lambda x: x
        self.optimizer = None

    def configure_optimizers(self):
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_scheduler(self, scheduler, scheduler_config):
        self.scheduler = scheduler_config
        self.scheduler["scheduler"] = scheduler

    def forward(self, x_a=None, x_b=None):
        if x_a is not None:
            emb_a_ = self.encoder_a(x_a)
            if self.h_a is None:
                self.h_a = MultiLayerPerceptron(
                    num_input_features=emb_a_.size(-1),
                    hidden_layer_dimensions=[self.dim],
                ).to(emb_a_)
            emb_a = F.normalize(self.h_a(self.proj_func(emb_a_)))  # [B, F]

        if x_b is not None:
            emb_b = self.encoder_b(x_b)
            if self.h_b is None:
                self.h_b = MultiLayerPerceptron(
                    num_input_features=emb_b.size(-1),
                    hidden_layer_dimensions=[self.dim],
                ).to(emb_b)
            emb_b = F.normalize(self.h_b(self.proj_func(emb_b)))  # [B, F]
        logits = torch.matmul(emb_a, emb_b.T)  # [B, B]

        if self.supervised_head is None:
            self.supervised_head = MultiLayerPerceptron(
                num_input_features=emb_a_.size(-1),
                hidden_layer_dimensions=self.supervised_head_dim,
            )
        supervised_logits = self.supervised_head(emb_a_)
        return logits, supervised_logits

    def _step(self, batch, batch_idx, step_name, dataloader_idx=None):
        inputs = batch["inputs"]
        supervised_labels = batch["labels"]
        logits, supervised_logits = self.forward(**inputs)

        if step_name == "train":
            logits = logits * self.temperature
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        labels = torch.arange(len(logits)).to(device=logits.device).long()

        loss_a = criterion(logits, labels)
        loss_b = criterion(logits.T, labels)
        loss = (loss_a + loss_b) / 2
        total = len(logits)

        logs = {}
        mask = supervised_labels != -1
        if mask.sum() != 0:
            mask = supervised_labels != -1
            supervised_criterion = nn.BCEWithLogitsLoss(reduction="none")
            supervised_loss = supervised_criterion(supervised_logits, supervised_labels)
            supervised_loss = torch.masked_select(supervised_loss, mask).mean()
            loss += supervised_loss

            supervised_outputs = torch.sigmoid(supervised_logits)

            logs.update(_supervised_metric(supervised_labels, supervised_outputs))
            logs["supervised_loss"] = float(supervised_loss.cpu().detach().item())

        logits = logits.detach()
        labels = labels.detach()

        topk = (1, 5, 10)
        acc_a = accuracy(logits.detach(), labels.detach(), topk=topk)
        acc_b = accuracy(logits.detach().t(), labels.detach(), topk=topk)
        for k, acc_ak, acc_bk in zip(topk, acc_a, acc_b):
            suffix = f"_top{k}" if k != 1 else ""
            acc_ak = acc_ak.cpu().item()
            acc_bk = acc_bk.cpu().item()
            logs.update(
                {
                    f"acc_a{suffix}": float(acc_ak),
                    f"acc_b{suffix}": float(acc_bk),
                    f"acc{suffix}": float((acc_ak + acc_bk) / 2),
                }
            )

        correct_a = logits.detach().argmax(dim=1).eq(labels).sum().cpu().item()
        correct_b = logits.detach().argmax(dim=0).eq(labels).sum().cpu().item()
        logs.update(
            {
                "loss": float(loss.cpu().detach().item()),
                "loss_a": float(loss_a.cpu().detach().item()),
                "loss_b": float(loss_b.cpu().detach().item()),
                "acc_a_old": float(correct_a / total),
                "acc_b_old": float(correct_b / total),
                "acc_old": float((correct_a + correct_b) / 2 / total),
            }
        )

        logs = {f"{step_name}/{k}": v for k, v in logs.items()}

        batch_dictionary = {
            "loss": loss,
            "log": logs,
        }
        if step_name == "val" and "supervised_loss" in logs:
            batch_dictionary["outputs"] = supervised_outputs
            batch_dictionary["labels"] = supervised_labels
        return batch_dictionary

    def training_step(self, train_batch, batch_idx):
        batch_dictionary = self._step(
            batch=train_batch, batch_idx=batch_idx, step_name="train"
        )
        # Log all metrics including learning rate
        for k, v in batch_dictionary["log"].items():
            self.log(k, v, on_step=False, on_epoch=True, sync_dist=True)
        
        # Separately log current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.log('train/lr', current_lr, on_step=True, on_epoch=False, sync_dist=True)
        return batch_dictionary

    def validation_step(self, val_batch, batch_idx, dataloader_idx=None):
        batch_dict = self._step(
            batch=val_batch,
            batch_idx=batch_idx,
            step_name="val",
            dataloader_idx=dataloader_idx,
        )
        self.validation_step_outputs.append(batch_dict)
        return batch_dict

    def on_validation_epoch_end(self):
        self.validation_step_outputs.clear()


class LightningGGNNRegression(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = GatedGraphNeuralNetwork(**kwargs)
        self.optimizer = None
        self.scheduler = None
        self.validation_step_outputs = []

    def configure_optimizers(self):
        optimizers = {"optimizer": self.optimizer}
        if self.scheduler is not None:
            optimizers["lr_scheduler"] = self.scheduler
        return optimizers

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_scheduler(self, scheduler, scheduler_config):
        self.scheduler = scheduler_config
        self.scheduler["scheduler"] = scheduler

    def _step(self, batch, batch_idx, step_name, dataloader_idx=None):
        inputs = batch["inputs"]
        supervised_labels = batch["labels"]
        logits = self.model.forward(**inputs)
        criterion = nn.MSELoss()

        logs = {}
        supervised_loss = criterion(logits, supervised_labels)
        loss = supervised_loss

        logs["supervised_loss"] = float(supervised_loss.cpu().detach().item())

        logs.update({"loss": float(loss.cpu().detach().item())})

        logs = {f"{step_name}/{k}": v for k, v in logs.items()}

        batch_dictionary = {
            "loss": loss,
            "log": logs,
        }
        if step_name == "val":
            batch_dictionary["outputs"] = logits
            batch_dictionary["labels"] = supervised_labels
        return batch_dictionary

    def training_step(self, train_batch, batch_idx):
        batch_dictionary = self._step(
            batch=train_batch, batch_idx=batch_idx, step_name="train"
        )

        for k, v in batch_dictionary["log"].items():
            self.log(k, v, on_step=False, on_epoch=True, sync_dist=True)

        return batch_dictionary

    def validation_step(self, val_batch, batch_idx, dataloader_idx=None):
        batch_dict = self._step(
            batch=val_batch,
            batch_idx=batch_idx,
            step_name="val",
            dataloader_idx=dataloader_idx,
        )
        self.validation_step_outputs.append(batch_dict)
        return batch_dict

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def on_validation_epoch_end(self):
        _validation_epoch_end(self, self.validation_step_outputs, is_regression=True)
        self.validation_step_outputs.clear()


class LightningGGNN(pl.LightningModule):
    def __init__(self, freeze=False, loss_type="bce", focal_gamma=2.0, **kwargs):
        super().__init__()
        self.model = GatedGraphNeuralNetwork(**kwargs)
        self.optimizer = None
        self.scheduler = None
        self.validation_step_outputs = []
        
        # Initialize loss function once in __init__
        if loss_type == "focal":
            self.supervised_criterion = FocalLoss(gamma=focal_gamma)
        else:  # Default to BCE
            self.supervised_criterion = nn.BCEWithLogitsLoss(reduction="none")
            
        if freeze:
            self.model.transfer(freeze=True)

    def configure_optimizers(self):
        optimizers = {"optimizer": self.optimizer}
        if self.scheduler is not None:
            optimizers["lr_scheduler"] = self.scheduler
        return optimizers

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_scheduler(self, scheduler, scheduler_config):
        self.scheduler = scheduler_config
        self.scheduler["scheduler"] = scheduler

    def _step(self, batch, batch_idx, step_name, dataloader_idx=None):
        inputs = batch["inputs"]
        supervised_labels = batch["labels"]
        logits = self.model.forward(**inputs)

        logs = {}
        mask = supervised_labels != -1
        if mask.sum() != 0:
            # Use pre-initialized loss function
            supervised_loss = self.supervised_criterion(logits, supervised_labels)
            supervised_loss = torch.masked_select(supervised_loss, mask).mean()
            loss = supervised_loss

            supervised_outputs = torch.sigmoid(logits)

            logs.update(_supervised_metric(supervised_labels, supervised_outputs))
            logs["supervised_loss"] = float(supervised_loss.cpu().detach().item())

        logs.update({"loss": float(loss.cpu().detach().item())})

        logs = {f"{step_name}/{k}": v for k, v in logs.items()}

        batch_dictionary = {
            "loss": loss,
            "log": logs,
        }
        if step_name == "val":
            batch_dictionary["outputs"] = supervised_outputs
            batch_dictionary["labels"] = supervised_labels
        return batch_dictionary

    def training_step(self, train_batch, batch_idx):
        batch_dictionary = self._step(
            batch=train_batch, batch_idx=batch_idx, step_name="train"
        )

        for k, v in batch_dictionary["log"].items():
            self.log(k, v, on_step=False, on_epoch=True, sync_dist=True)

        return batch_dictionary

    def validation_step(self, val_batch, batch_idx, dataloader_idx=None):
        batch_dict = self._step(
            batch=val_batch,
            batch_idx=batch_idx,
            step_name="val",
            dataloader_idx=dataloader_idx,
        )
        self.validation_step_outputs.append(batch_dict)
        return batch_dict

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def on_train_epoch_start(self):
        self.log("learning_rate_epoch", self.optimizer.param_groups[0]["lr"], on_step=False, on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self):
        _validation_epoch_end(self, self.validation_step_outputs)
        self.validation_step_outputs.clear()


class GatedGraphNeuralNetwork(nn.Module):
    """A variant of the graph neural network family that utilizes GRUs
    to control the flow of information between layers.
    Each GatedGraphConvolution operation follows the formulations below:
    .. math:: H^{(L_i)} = A H^{(L-1)} W^{(L)}
    .. math:: H^{(L)} = GRU(H^{(L-1)}, H^{(L_i)})
    The current implementation also facilites transfer learning with the "transfer" method.
    The method replaces the last fully connected layer in the trained model object
    with a reinitialized layer that has a specified output dimension.
    Example:
    >>> # here we instantiate a model with output dimension 1
    >>> model = GatedGraphNeuralNetwork(n_edge=1, in_dim=10, n_conv=5, fc_dims=[1024, 1])
    >>> # now we reinitializes the last layer to have output dimension of 50
    >>> model.transfer(out_dim=50)
    Gated Graph Sequence Neural Networks: https://arxiv.org/abs/1511.05493
    Neural Message Passing for Quantum Chemistry: https://arxiv.org/abs/1704.01212
    """

    def __init__(
        self,
        n_edge: int,
        in_dim: int,
        n_conv: int,
        fc_dims: Iterable[int],
        p_dropout: float = 0.2,
    ) -> None:
        """Gated graph neural network with support for transfer learning
        Parameters
        ----------
        n_edge : int
            Number of edges in input graphs.
        in_dim : int
            Number of features per node in input graphs.
        n_conv : int
            Number of gated graph convolution layers.
        fc_dims : Iterable[int]
            Fully connected layers dimensions.
        """
        super(GatedGraphNeuralNetwork, self).__init__()

        self.conv_layers, self.fc_layers = self._build_layers(
            in_dim=in_dim, n_edge=n_edge, fc_dims=fc_dims, n_conv=n_conv
        )

        self.dropout = nn.Dropout(p=p_dropout)
        self.reset_parameters()

    @staticmethod
    def _build_layers(in_dim, n_edge, fc_dims, n_conv):
        conv_layers = []

        for i in range(n_conv):
            l = GatedGraphConvolution(in_dim=in_dim, out_dim=in_dim, n_edge=n_edge)
            conv_layers.append(l)

        fc_layers = []
        num_fc_layers = len(fc_dims)
        fc_dims.insert(0, in_dim)
        for i, (in_dim, out_dim) in enumerate(zip(fc_dims[:-1], fc_dims[1:])):
            l = nn.Linear(in_dim, out_dim)

            if i < (num_fc_layers - 2):
                l = nn.Sequential(l, nn.ReLU())
            elif i == (num_fc_layers - 2):
                l = nn.Sequential(l, nn.Tanh())

            fc_layers.append(l)

        return nn.ModuleList(conv_layers), nn.ModuleList(fc_layers)

    def reset_parameters(self):
        for l in self.conv_layers:
            l.reset_parameters()

        for k, v in self.state_dict().items():
            if "fc_layers" in k:
                if "weight" in k:
                    nn.init.xavier_uniform_(v)
                elif "bias" in k:
                    nn.init.zeros_(v)

    def encode(self, x: List[torch.Tensor]) -> torch.Tensor:
        """Encode featurized batched input.
        This is done by forward propagating up to the second to last layer in the network.
        Parameters
        ----------
        x : List[torch.Tensor]
            List of batch input torch.Tensor [adj_mat, node_feat, atom_vec ]
        Returns
        -------
        torch.Tensor
            Encoded inputs
        """
        adj, node_feat, atom_vec = x

        for layer in self.conv_layers:
            node_feat = layer(adj, node_feat)
            node_feat = self.dropout(node_feat)
        output = torch.mul(node_feat, atom_vec)

        output = output.sum(1)

        for layer in self.fc_layers[:-1]:
            output = layer(output)
            output = self.dropout(output)

        return output

    def forward(self, x_a: List[torch.Tensor], **kwargs) -> torch.Tensor:
        """Run forward pass on batched input.
        Parameters
        ----------
        x : List[torch.Tensor]
            List of batch input torch.Tensor [adj_mat, node_feat, atom_vec]
        Returns
        -------
        torch.Tensor
            Model output
        """
        output = self.encode(x_a)
        output = self.fc_layers[-1](output)
        return output

    def transfer(self, out_dim: Union[list, int] = None, freeze: bool = False) -> None:
        """Replace the last fully connected layer with a newly initialized layer
        with out_dim as output dimension. Use freeze=True to freeze the pre-trained
        network and use it as a featurizer.
        Parameters
        ----------
        out_dim : Union[list,int]
            Output dimension of the new fully connected layer
        freeze : bool, optional
            Freeze the weights of the pretrained network, by default False
        """
        # only transfer learn on graph level
        self.dropout = nn.Dropout(p=0.1)

        # freeze parameters if necessary
        if freeze:
            for param in self.parameters():
                param.requires_grad = False

        # new final fc layer overriding freeze
        if out_dim is None:
            out_dim = self.fc_layers[-1].out_features

        if isinstance(out_dim, int):
            out_dim = [out_dim]

        in_dim = self.fc_layers[-1].in_features
        out_dim.insert(0, in_dim)
        del self.fc_layers[-1]

        for i, (in_dim, out_dim_) in enumerate(zip(out_dim[:-1], out_dim[1:])):
            layer = nn.Linear(in_dim, out_dim_)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.fc_layers.append(layer)

        return None


class TripleInputEncoder(pl.LightningModule):
    def __init__(
        self,
        encoder_a: Optional[MultiLayerPerceptron],  # molecular encoder
        encoder_b: Optional[MultiLayerPerceptron],  # morphological encoder
        encoder_c: Optional[MultiLayerPerceptron],  # genomic encoder
        supervised_head_dim=[64, 2],
        non_lin_proj: bool = False,
        dim=128,
        temperature=10,
    ):
        super().__init__()
        self.encoder_a = encoder_a
        self.encoder_b = encoder_b
        self.encoder_c = encoder_c
        self.dim = dim
        self.supervised_head_dim = supervised_head_dim
        self.add_module("supervised_head", None)
        self.add_module("h_a", None)
        self.add_module("h_b", None)
        self.add_module("h_c", None)
        self.temperature = temperature
        self.validation_step_outputs = []

        if non_lin_proj:
            self.proj_func = F.relu
        else:
            self.proj_func = lambda x: x
        self.optimizer = None

    def forward(self, x_a=None, x_b=None, x_c=None):
        # Encode molecular features
        if x_a is not None:
            emb_a_ = self.encoder_a(x_a)
            if self.h_a is None:
                self.h_a = MultiLayerPerceptron(
                    num_input_features=emb_a_.size(-1),
                    hidden_layer_dimensions=[self.dim],
                ).to(emb_a_)
            emb_a = F.normalize(self.h_a(self.proj_func(emb_a_)))  # [B, F]

        # Encode morphological features
        if x_b is not None:
            emb_b = self.encoder_b(x_b)
            if self.h_b is None:
                self.h_b = MultiLayerPerceptron(
                    num_input_features=emb_b.size(-1),
                    hidden_layer_dimensions=[self.dim],
                ).to(emb_b)
            emb_b = F.normalize(self.h_b(self.proj_func(emb_b)))  # [B, F]

        # Encode genomic features
        if x_c is not None:
            emb_c = self.encoder_c(x_c)
            if self.h_c is None:
                self.h_c = MultiLayerPerceptron(
                    num_input_features=emb_c.size(-1),
                    hidden_layer_dimensions=[self.dim],
                ).to(emb_c)
            emb_c = F.normalize(self.h_c(self.proj_func(emb_c)))  # [B, F]

        # Calculate contrastive logits
        logits_ab = torch.matmul(emb_a, emb_b.T)  # mol-morph
        logits_ac = torch.matmul(emb_a, emb_c.T)  # mol-genomic

        if self.supervised_head is None:
            self.supervised_head = MultiLayerPerceptron(
                num_input_features=emb_a_.size(-1),
                hidden_layer_dimensions=self.supervised_head_dim,
            )
        supervised_logits = self.supervised_head(emb_a_)
        
        return logits_ab, logits_ac, supervised_logits

    def _step(self, batch, batch_idx, step_name, dataloader_idx=None):
        inputs = batch["inputs"]
        supervised_labels = batch["labels"]
        logits_ab, logits_ac, supervised_logits = self.forward(**inputs)

        # Create masks for valid samples (not all -1s)
        b_valid = ~torch.all(inputs['x_b'] == -1, dim=1)
        c_valid = ~torch.all(inputs['x_c'] == -1, dim=1)
        
        # Apply temperature scaling
        if step_name == "train":
            logits_ab = logits_ab * self.temperature
            logits_ac = logits_ac * self.temperature
            
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        labels = torch.arange(len(logits_ab)).to(device=logits_ab.device).long()
        
        # Calculate losses for valid pairs
        loss = 0
        n_valid_losses = 0
        
        if b_valid.any():
            loss_ab = criterion(logits_ab[b_valid], labels[b_valid])
            loss_ba = criterion(logits_ab.T[b_valid], labels[b_valid])
            loss += loss_ab + loss_ba
            n_valid_losses += 2
        else:
            loss_ab = loss_ba = torch.tensor(0.0, device=logits_ab.device)
        
        if c_valid.any():
            loss_ac = criterion(logits_ac[c_valid], labels[c_valid])
            loss_ca = criterion(logits_ac.T[c_valid], labels[c_valid])
            loss += loss_ac + loss_ca
            n_valid_losses += 2
        else:
            loss_ac = loss_ca = torch.tensor(0.0, device=logits_ab.device)
        
        loss = loss / max(n_valid_losses, 1)

        logs = {}
        # Handle supervised loss if present
        mask = supervised_labels != -1
        if mask.sum() != 0:
            supervised_criterion = nn.BCEWithLogitsLoss(reduction="none")
            supervised_loss = supervised_criterion(supervised_logits, supervised_labels)
            supervised_loss = torch.masked_select(supervised_loss, mask).mean()
            loss += supervised_loss

            supervised_outputs = torch.sigmoid(supervised_logits)
            logs.update(_supervised_metric(supervised_labels, supervised_outputs))
            logs["supervised_loss"] = float(supervised_loss.cpu().detach().item())

        # Calculate accuracies using DualInputEncoder style
        topk = (1, 5, 10)
        acc_a = accuracy(logits_ab.detach(), labels.detach(), topk=topk)
        acc_b = accuracy(logits_ab.detach().t(), labels.detach(), topk=topk)
        
        for k, acc_ak, acc_bk in zip(topk, acc_a, acc_b):
            suffix = f"_top{k}" if k != 1 else ""
            acc_ak = acc_ak.cpu().item()
            acc_bk = acc_bk.cpu().item()
            logs.update({
                f"acc_a{suffix}": float(acc_ak),
                f"acc_b{suffix}": float(acc_bk),
                f"acc{suffix}": float((acc_ak + acc_bk) / 2),
            })

        logs.update({
            "loss": float(loss.cpu().detach().item()),
            "loss_ab": float(loss_ab.cpu().detach().item()),
            "loss_ba": float(loss_ba.cpu().detach().item()),
            "loss_ac": float(loss_ac.cpu().detach().item()),
            "loss_ca": float(loss_ca.cpu().detach().item()),
            "n_valid_morph": float(b_valid.sum().item()),
            "n_valid_genomic": float(c_valid.sum().item()),
            "n_valid_losses": float(n_valid_losses)
        })


        logs = {f"{step_name}/{k}": v for k, v in logs.items()}

        batch_dictionary = {
            "loss": loss,
            "log": logs,
        }

        if step_name == "val" and "supervised_loss" in logs:
            batch_dictionary["outputs"] = supervised_outputs
            batch_dictionary["labels"] = supervised_labels

        return batch_dictionary

    # Keep the rest of the methods same as DualInputEncoder
    def configure_optimizers(self):
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_scheduler(self, scheduler, scheduler_config):
        self.scheduler = scheduler_config
        self.scheduler["scheduler"] = scheduler

    def training_step(self, train_batch, batch_idx):
        batch_dictionary = self._step(
            batch=train_batch, batch_idx=batch_idx, step_name="train"
        )
        for k, v in batch_dictionary["log"].items():
            self.log(k, v, on_step=False, on_epoch=True, sync_dist=True)
        return batch_dictionary

    def validation_step(self, val_batch, batch_idx, dataloader_idx=None):
        batch_dict = self._step(
            batch=val_batch,
            batch_idx=batch_idx,
            step_name="val",
            dataloader_idx=dataloader_idx,
        )
        self.validation_step_outputs.append(batch_dict)
        return batch_dict

    def on_validation_epoch_end(self):
        _validation_epoch_end(self, self.validation_step_outputs)
        self.validation_step_outputs.clear()

class CellLineTripleInputEncoder(pl.LightningModule):
    def __init__(
        self,
        encoder_a: Optional[MultiLayerPerceptron],
        encoder_b: Optional[MultiLayerPerceptron],
        encoder_c: Optional[MultiLayerPerceptron],
        # Pretrained encoder paths
        pretrained_encoder_b_path: Optional[str] = None,
        pretrained_encoder_c_path: Optional[str] = None,
        # Pretrained encoder options
        freeze_encoder_b: bool = False,
        freeze_encoder_c: bool = False,
        use_latent_layer_b: bool = True,
        use_latent_layer_c: bool = True,
        load_pretrained_weights_b: bool = True,
        load_pretrained_weights_c: bool = True,
        # Cell line embedding parameters
        n_cell_lines: int = 24,
        n_dose_levels: int = 6,
        n_time_points: int = 2,
        cell_embedding_dim: int = 32,
        dose_embedding_dim: int = 32,
        time_embedding_dim: int = 32,
        # Other parameters
        temperature=10,
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        super().__init__()
        self.encoder_a = encoder_a
        
        # Load pretrained encoders if paths provided, otherwise use provided encoders
        if pretrained_encoder_b_path is not None:
            print(f"Loading pretrained encoder_b from: {pretrained_encoder_b_path}")
            self.encoder_b = load_pretrained_encoder(
                checkpoint_path=pretrained_encoder_b_path,
                freeze=freeze_encoder_b,
                device=device,
                load_weights=load_pretrained_weights_b
            )
            print(f"Pretrained encoder_b loaded (frozen={freeze_encoder_b})")
        else:
            self.encoder_b = encoder_b
            
        if pretrained_encoder_c_path is not None:
            print(f"Loading pretrained conditional genomic encoder from: {pretrained_encoder_c_path}")
            self.encoder_c = load_pretrained_encoder(
                checkpoint_path=pretrained_encoder_c_path,
                freeze=freeze_encoder_c,
                device=device,
                load_weights=load_pretrained_weights_c
            )
            print(f"Pretrained conditional genomic encoder loaded (frozen={freeze_encoder_c})")
            print("Note: Pretrained encoder expects genomic features + conditional embeddings as input")
        else:
            # Store genomic encoder - expects concatenated input: genomic_features + cell_embedding + dose_embedding + time_embedding
            # This applies to both pretrained and original encoders now
            self.encoder_c = encoder_c
        
        # Rest of initialization
        self.temperature = temperature
        
        # Cell line embedding with 0 as padding
        self.cell_embedding = nn.Embedding(
            num_embeddings=n_cell_lines + 1,  # +1 because indices start from 1
            embedding_dim=cell_embedding_dim,
            padding_idx=0  # Use 0 as padding
        )
        
        # Dose level embedding with -1 as padding
        self.dose_embedding = nn.Embedding(
            num_embeddings=n_dose_levels + 1,  # +1 for padding
            embedding_dim=dose_embedding_dim,
            padding_idx=0  # Use 0 as padding
        )
        
        # Time point embedding with -1 as padding
        self.time_embedding = nn.Embedding(
            num_embeddings=n_time_points + 1,  # +1 for padding
            embedding_dim=time_embedding_dim,
            padding_idx=0  # Use 0 as padding
        )
        
        self.optimizer = None
        self.validation_step_outputs = []
        
        # Store sample batches but exclude from state_dict to avoid pickling errors
        # These contain molecular graphs that can't be pickled
        self.register_buffer('_sample_train_batch', torch.tensor([]), persistent=False)
        self.register_buffer('_sample_val_batch', torch.tensor([]), persistent=False)
        # Use Python attributes for actual storage (not pickled)
        self._train_batch_cache = None
        self._val_batch_cache = None

    def forward(self, x_a=None, x_b=None, x_c=None, cell_indices=None, doses=None, times=None, batch_indices=None):
        """Forward pass with efficient genomic data processing."""
        # 1. Encode molecular features
        if x_a is not None:
            emb_a = F.normalize(self.encoder_a(x_a))  # [B, 128] - GNN now outputs 128-dim directly

        # 2. Encode morphological features
        emb_b = None
        if x_b is not None:
            # Check for valid samples (not all -1s)
            b_valid = ~torch.all(x_b == -1, dim=1)
            
            if b_valid.any():
                # Process only valid samples through encoder
                emb_b = F.normalize(self.encoder_b(x_b[b_valid]))  # [n_valid, 128]

        # 3. Encode genomic features (only if we have valid conditions)
        emb_c = None
        if x_c is not None and len(x_c) > 0:
            # Always concatenate genomic features with conditional embeddings
            # This is needed for both pretrained and original encoders now
            cell_emb = self.cell_embedding(cell_indices.long())  # [total_conditions, E_cell]
            dose_emb = self.dose_embedding(doses.long())  # [total_conditions, E_dose]
            time_emb = self.time_embedding(times.long())  # [total_conditions, E_time]
            
            # Concatenate genomic features with all embeddings
            x_c_combined = torch.cat([
                x_c,  # [total_conditions, G]
                cell_emb,  # [total_conditions, E_cell]
                dose_emb,  # [total_conditions, E_dose]
                time_emb   # [total_conditions, E_time]
            ], dim=-1)  # [total_conditions, G+E_cell+E_dose+E_time]
               
            # Process through genomic encoder (pretrained or original) and normalize
            emb_c = F.normalize(self.encoder_c(x_c_combined))  # [total_conditions, 128] - outputs 128-dim

        # 4. Calculate contrastive logits (standard bidirectional approach)
        # Morphological contrastive logits
        logits_ab = None
        if emb_b is not None:
            logits_ab = torch.matmul(emb_a, emb_b.T)  # [B, B]
        
        # Genomic contrastive logits
        logits_ac = None
        if emb_c is not None and len(emb_c) > 0:
            # Expand molecular embeddings to match genomic conditions
            # batch_indices tells us which sample each genomic condition belongs to
            emb_a_expanded = emb_a[batch_indices]  # [total_conditions, F]
            
            # Compute similarity matrix between expanded molecular and genomic embeddings
            logits_ac = torch.matmul(emb_a_expanded, emb_c.T)  # [total_conditions, total_conditions]
        
        return logits_ab, logits_ac, emb_a, emb_b, emb_c

    def _compute_contrastive_losses(self, logits_ab, logits_ac, emb_a, emb_b, emb_c, inputs, step_name):
        """Compute bidirectional contrastive losses for morphological and genomic modalities."""
        loss = 0
        n_valid_losses = 0
        logs = {}
        
        # Apply temperature scaling
        if step_name == "train":
            if logits_ab is not None:
                logits_ab = logits_ab * self.temperature
            if logits_ac is not None:
                logits_ac = logits_ac * self.temperature
        
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        
        # 1. Morphological contrastive loss (bidirectional like DualInputEncoder)
        if logits_ab is not None:
            b_valid = ~torch.all(inputs['x_b'] == -1, dim=1)
            batch_size = logits_ab.size(0)
            
            # Create labels: valid samples get their index, invalid samples get -1 (ignored)
            labels = torch.full((batch_size,), -1, dtype=torch.long, device=logits_ab.device)
            labels[b_valid] = torch.arange(b_valid.sum(), device=logits_ab.device)
            
            # Forward: mol->morph (invalid samples are ignored due to ignore_index=-1)
            loss_ab = criterion(logits_ab, labels)
            
            # Backward: morph->mol (only compute for valid morphological samples)
            if b_valid.any():
                # For backward loss, we need to map each morphological sample to its corresponding molecular sample
                # logits_ab.T has shape [n_valid, batch_size]
                # We need labels of size [n_valid] that point to the correct molecular sample
                valid_mol_indices = torch.where(b_valid)[0]  # Which molecular samples are valid
                loss_ba = criterion(logits_ab.T, valid_mol_indices)
            else:
                loss_ba = torch.tensor(0.0, device=logits_ab.device)
            
            morphological_loss = (loss_ab + loss_ba) / 2
            loss += morphological_loss
            n_valid_losses += 1

            logs.update({
                "morphological_loss": float(morphological_loss.cpu().detach().item()),
                "loss_ab": float(loss_ab.cpu().detach().item()),
                "loss_ba": float(loss_ba.cpu().detach().item()),
            })
            
            # Check for collapse in morphological embeddings using normalized logits
            if step_name == "train" and b_valid.any():
                # Use softmax normalized logits for collapse detection (probabilities)
                probs = F.softmax(logits_ab, dim=1)
                
                # Create labels for morphological data (diagonal = positive, off-diagonal = negative)
                morph_labels = torch.eye(logits_ab[b_valid].size(0), device=logits_ab.device)
                check_collapse(emb_b, probs[b_valid], self, name="morph", labels=morph_labels)
        
        # 2. Genomic contrastive loss using multi-positive InfoNCE:
        # Elegant approach that handles multiple genomic conditions per molecule naturally
        if logits_ac is not None and logits_ac.size(0) > 0:
            batch_indices = inputs['batch_indices']
            n_conditions = logits_ac.size(0)
            
            # Get molecule IDs for each condition
            mol_ids = batch_indices  # [total_conditions] - which molecule each condition belongs to
            
            # Step 1: Similarity matrix (already computed as logits_ac)
            # logits_ac = emb_a_expanded @ emb_c.T / temperature
            
            # Step 2: Positive labels - conditions belong to same molecule are positives
            labels = (mol_ids[:, None] == mol_ids[None, :]).float()  # [n_conditions, n_conditions]
            
            # Step 3: Forward loss (molecular → genomic)
            probs_fwd = F.softmax(logits_ac, dim=1)  # [n_conditions, n_conditions]
            pos_probs_fwd = (probs_fwd * labels).sum(dim=1)  # [n_conditions] - sum of positive probabilities per row
            loss_fwd = -torch.log(pos_probs_fwd + 1e-8).mean()  # Multi-positive InfoNCE
            
            # Step 4: Backward loss (genomic → molecular)
            probs_bwd = F.softmax(logits_ac.T, dim=1)  # [n_conditions, n_conditions]
            pos_probs_bwd = (probs_bwd * labels.T).sum(dim=1)  # [n_conditions] - sum of positive probabilities per row
            loss_bwd = -torch.log(pos_probs_bwd + 1e-8).mean()  # Multi-positive InfoNCE
            
            # Step 5: Final symmetric contrastive loss
            final_genomic_loss = (loss_fwd + loss_bwd) / 2
            loss += final_genomic_loss
            n_valid_losses += 1
            
            # Get molecule statistics for logging
            unique_batch_indices, inverse_indices = torch.unique(batch_indices, return_inverse=True)
            n_molecules_with_genomic = len(unique_batch_indices)
            
            logs.update({
                "genomic_loss": float(final_genomic_loss.cpu().detach().item()),
                "loss_ac": float(loss_fwd.cpu().detach().item()), 
                "loss_ca": float(loss_bwd.cpu().detach().item()),
                "n_genomic_conditions": float(n_conditions),
                "n_molecules_with_genomic": float(n_molecules_with_genomic),
                "avg_conditions_per_molecule": float(n_conditions / max(n_molecules_with_genomic, 1)),
            })
            
            # Check for collapse in genomic embeddings using normalized logits
            if step_name == "train":
                # Use softmax normalized logits for collapse detection (probabilities)
                probs = F.softmax(logits_ac, dim=1)
                check_collapse(emb_c, probs, self, name="genomic", labels=labels)
        else:
            # When no genomic data is available, only include the count metrics
            logs.update({
                "n_genomic_conditions": 0.0,
                "n_molecules_with_genomic": 0.0,
                "avg_conditions_per_molecule": 0.0
            })
        
        # Average loss over number of modalities
        loss = loss / max(n_valid_losses, 1)
        logs["n_valid_losses"] = float(n_valid_losses)
        
        return loss, logs



    def _compute_accuracies(self, logits_ab, logits_ac, inputs):
        """Compute top-k accuracies for morphological and genomic modalities."""
        logs = {}
        topk = (1, 5)
        
        # Morphological accuracies (bidirectional)
        if logits_ab is not None:
            b_valid = ~torch.all(inputs['x_b'] == -1, dim=1)
            if b_valid.any():
                batch_size = logits_ab.size(0)
                
                # Create labels: valid samples get their index, invalid samples get -1 (ignored)
                labels = torch.full((batch_size,), -1, dtype=torch.long, device=logits_ab.device)
                labels[b_valid] = torch.arange(b_valid.sum(), device=logits_ab.device)
                
                # b_valid selected the rows (molecules) that are valid
                acc_ab = accuracy(logits_ab[b_valid].detach(), labels[b_valid].detach(), topk=topk)
                 # b_valid selected the columns (molecules) that are valid
                acc_ba = accuracy(logits_ab.T[:,b_valid].detach(), labels[b_valid].detach(), topk=topk)
                
                for k, acc_ak, acc_bk in zip(topk, acc_ab, acc_ba):
                    suffix = f"_top{k}" if k != 1 else ""
                    acc_ak = acc_ak.cpu().item()
                    acc_bk = acc_bk.cpu().item()
                    logs.update({
                        f"morph_acc_a{suffix}": float(acc_ak),
                        f"morph_acc_b{suffix}": float(acc_bk),
                        f"morph_acc{suffix}": float((acc_ak + acc_bk) / 2),
                    })
        
        # Genomic accuracies (molecular-genomic alignment)
        if logits_ac is not None and logits_ac.size(0) > 0:
            batch_indices = inputs['batch_indices']
            
            # Compute vectorized genomic accuracy: "Given a genomic condition, can we find its molecule?"
            genomic_accuracy = self._compute_genomic_accuracy(logits_ac.detach(), batch_indices)
            
            # Since forward and backward are the same metric, use the same value for both
            logs.update({
                "genomic_acc_a": float(genomic_accuracy),
                "genomic_acc_c": float(genomic_accuracy),
                "genomic_acc": float(genomic_accuracy),
            })
        else:
            # When no genomic data is available, don't include genomic accuracy metrics
            pass
        
        return logs

    def _compute_genomic_accuracy(self, logits: torch.Tensor, batch_indices: torch.Tensor) -> float:
        """
        Compute accuracy for contrastive setup:
        Given similarity logits [N, N] and batch_indices (molecule IDs),
        accuracy is the fraction of rows where the argmax prediction
        corresponds to one of the positives for that row.

        Args:
            logits: Tensor [N, N] similarity matrix
            batch_indices: Tensor [N] with molecule IDs (same ID = positive pair)

        Returns:
            accuracy: float in [0,1]
        """
        # Positive mask (same molecule ID = positive)
        mask = (batch_indices[:, None] == batch_indices[None, :])

        # Predicted column (argmax)
        preds = logits.argmax(dim=1)

        # Check if argmax is positive
        correct = mask[torch.arange(len(preds)), preds]
        correct = correct.bool().float()
        
        acc = correct.mean().item()
        return acc


    def _step(self, batch, batch_idx, step_name, dataloader_idx=None):
        inputs = batch["inputs"]
        
        # Forward pass
        logits_ab, logits_ac, emb_a, emb_b, emb_c = self.forward(**inputs)
        
        # Compute contrastive losses
        contrastive_loss, contrastive_logs = self._compute_contrastive_losses(
            logits_ab, logits_ac, emb_a, emb_b, emb_c, inputs, step_name
        )
        
        # Compute accuracies
        accuracy_logs = self._compute_accuracies(logits_ab, logits_ac, inputs)
        
        # Combine all logs
        logs = {}
        logs.update(contrastive_logs)
        logs.update(accuracy_logs)
        logs["loss"] = contrastive_loss.detach()  # Keep as GPU tensor for sync_dist
        
        logs = {f"{step_name}/{k}": v for k, v in logs.items()}

        batch_dictionary = {
            "loss": contrastive_loss,
            "log": logs,
        }

        return batch_dictionary

    # Keep the rest of the methods same as TripleInputEncoder
    def configure_optimizers(self):
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_scheduler(self, scheduler, scheduler_config):
        self.scheduler = scheduler_config
        self.scheduler["scheduler"] = scheduler

    def training_step(self, train_batch, batch_idx):
        batch_dictionary = self._step(
            batch=train_batch, batch_idx=batch_idx, step_name="train"
        )
        for k, v in batch_dictionary["log"].items():
            self.log(k, v, on_step=False, on_epoch=True, sync_dist=True)

        # Store the first training batch for visualization (not checkpointed)
        if self._train_batch_cache is None:
            self._train_batch_cache = train_batch

        return batch_dictionary

    def validation_step(self, val_batch, batch_idx, dataloader_idx=None):
        # Store the first validation batch for visualization (not checkpointed)
        if self._val_batch_cache is None:
            self._val_batch_cache = val_batch
        batch_dict = self._step(
            batch=val_batch,
            batch_idx=batch_idx,
            step_name="val",
            dataloader_idx=dataloader_idx,
        )
        self.validation_step_outputs.append(batch_dict)
        return batch_dict
    
    def on_train_epoch_start(self):
        """Record the start time of each training epoch."""
        self.epoch_start_time = torch.cuda.Event(enable_timing=True)
        self.epoch_start_time.record()

        # Clear previous outputs and reset the sample batch cache
        self._train_batch_cache = None
        
        # Log memory usage
        log_memory_usage(self, "epoch_start")

    def on_train_epoch_end(self):
        """Log learning rate and epoch time at the end of training epoch."""
        # Record end time and calculate duration
        epoch_end_time = torch.cuda.Event(enable_timing=True)
        epoch_end_time.record()
        torch.cuda.synchronize()
        epoch_time_min = self.epoch_start_time.elapsed_time(epoch_end_time) / 60000.0  # Convert to minutes

        # Log metrics
        self.log("learning_rate_epoch", float(self.optimizer.param_groups[0]["lr"]), on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/epoch_time_min", float(epoch_time_min), on_step=False, on_epoch=True, sync_dist=True)
        
        # CPU memory cleanup to prevent accumulation from persistent workers
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_validation_epoch_end(self):
        """
        Gather validation outputs from all GPUs before computing metrics.
        This ensures we compute metrics on the complete validation set.
        """
        validation_step_outputs = self.validation_step_outputs
        
        # In PyTorch Lightning 2.0+, use self.trainer.world_size
        world_size = self.trainer.world_size
        
        if world_size > 1:
            # Multi-GPU: gather outputs from all GPUs
            
            # Ensure outputs are in list format
            if isinstance(validation_step_outputs, dict):
                validation_step_outputs = [validation_step_outputs]
            
            # Gather from all GPUs
            gathered_outputs = [None] * world_size
            dist.all_gather_object(gathered_outputs, validation_step_outputs)
            
            # Flatten: [[gpu0_batches], [gpu1_batches], ...] -> [all_batches]
            all_outputs = []
            for i, gpu_outputs in enumerate(gathered_outputs):
                if isinstance(gpu_outputs, list):
                    all_outputs.extend(gpu_outputs)
                else:
                    if gpu_outputs is not None:
                        all_outputs.append(gpu_outputs)
            
            # ALL ranks compute metrics on the complete validation set
            # This is necessary so all ranks have access to metrics for callbacks (e.g., early stopping)
            _validation_epoch_end(self, all_outputs)
            
            # Clean up gathered data to prevent memory accumulation
            del gathered_outputs, all_outputs
            gc.collect()
        else:
            # Single GPU: no gathering needed
            _validation_epoch_end(self, validation_step_outputs)
        
        # Clear outputs and validation batch cache for next epoch
        self.validation_step_outputs.clear()
        self._val_batch_cache = None


def check_collapse(embeddings, logits, model, name="emb", eps=1e-6, labels=None):
    """
    Check whether embeddings are collapsing and log metrics using logits matrices.
    
    Args:
        embeddings: [N, D] normalized embeddings (already normalized by F.normalize)
        logits: [N, N] logits matrix (normalized by F.softmax to probabilities)
        model: PyTorch Lightning model instance for logging
        name: prefix for logged metrics
        eps: threshold for variance-based collapse detection
        labels: [N, N] positive mask (1 for positive pairs, 0 for negative pairs)
    """
    with torch.no_grad():
        # 1. Variance per dimension (most important for collapse detection)
        var_per_dim = embeddings.var(dim=0)
        mean_var, min_var = var_per_dim.mean().item(), var_per_dim.min().item()

        # Use provided positive/negative labels for collapse detection
        pos_mask = labels.bool()
        neg_mask = ~pos_mask
        
        # Compute positive and negative similarities
        if pos_mask.any():
            pos_logits = logits[pos_mask]
            mean_logits_pos = pos_logits.mean().item()
            std_logits_pos = pos_logits.std().item()
        else:
            mean_logits_pos = 0.0
            std_logits_pos = 0.0
            
        if neg_mask.any():
            neg_logits = logits[neg_mask]
            mean_logits_neg = neg_logits.mean().item()
            std_logits_neg = neg_logits.std().item()
        else:
            mean_logits_neg = 0.0
            std_logits_neg = 0.0
        
        # Collapse detection: positive similarities should be higher than negative
        pos_neg_separation = mean_logits_pos - mean_logits_neg
        collapsed = (pos_neg_separation < 0.3) or (mean_var < eps)
        
        # Create log dictionary with positive/negative metrics
        log_dict = {
            f"{name}_mean_var": mean_var,
            f"{name}_min_var": min_var,
            f"{name}_mean_logits_pos": mean_logits_pos,
            f"{name}_std_logits_pos": std_logits_pos,
            f"{name}_mean_logits_neg": mean_logits_neg,
            f"{name}_std_logits_neg": std_logits_neg,
            f"{name}_pos_neg_separation": pos_neg_separation,
            f"{name}_collapsed": float(collapsed),
            f"{name}_collapse_warning": 1.0 if collapsed else 0.0,
        }
        
        
        # Log all metrics using the same structure as the rest of the codebase
        for k, v in log_dict.items():
            model.log(k, v, on_step=False, on_epoch=True, sync_dist=True)


class CellLineTripleInputEncoderSoftHard(CellLineTripleInputEncoder):
    """
    CellLineTripleInputEncoder with soft and hard contrastive losses for morphological modality.
    
    This class inherits from CellLineTripleInputEncoder and overrides the morphological
    contrastive loss computation to use:
    - Soft contrastive loss for forward direction (mol→morph)
    - Hard contrastive loss for backward direction (morph→mol)
    """
    
    def __init__(self, *args, gradient_stop=True, temperature_main=None, temperature_momentum=None, 
                 temperature_momentum_morph=None, temperature_momentum_genomic=None, **kwargs):
        """
        Initialize with optional gradient stopping for collapse prevention.
        
        Args:
            gradient_stop (bool): If True, apply stop gradient trick to target distributions (YY)
                                 to prevent collapse. If False, gradients flow through all distributions.
            temperature_main (float): Temperature for main encoder (student) - if None, uses parent's temperature
            temperature_momentum (float): Temperature for momentum encoder (teacher) - if None, uses parent's temperature
            temperature_momentum_morph (float): Temperature for morphological momentum encoder - if None, uses temperature_momentum
            temperature_momentum_genomic (float): Temperature for genomic momentum encoder - if None, uses temperature_momentum
        """
        super().__init__(*args, **kwargs)
        self.gradient_stop = gradient_stop
        
        # Set separate temperatures for main and momentum encoders
        self.temperature_main = temperature_main if temperature_main is not None else self.temperature
        self.temperature_momentum = temperature_momentum if temperature_momentum is not None else self.temperature
        
        # Set separate momentum temperatures for each modality
        self.temperature_momentum_morph = temperature_momentum_morph if temperature_momentum_morph is not None else self.temperature_momentum
        self.temperature_momentum_genomic = temperature_momentum_genomic if temperature_momentum_genomic is not None else self.temperature_momentum
        
        print(f"Temperature main: {self.temperature_main}")
        print(f"Temperature momentum: {self.temperature_momentum}")
        print(f"Temperature momentum (morphological): {self.temperature_momentum_morph}")
        print(f"Temperature momentum (genomic): {self.temperature_momentum_genomic}")
    
    
    def _compute_morphological_contrastive_loss(self, logits_XY, emb_b, is_training=True):
        """Compute morphological contrastive loss: soft forward + hard backward."""
        if logits_XY is None or emb_b is None:
            zero_loss = torch.tensor(0.0, device=logits_XY.device if logits_XY is not None else torch.device('cpu'))
            return zero_loss, zero_loss, zero_loss
        
        # Apply temperature scaling
        scale_main = self.temperature_main if is_training else 1.0
        scale_momentum = self.temperature_momentum_morph if is_training else 1.0
        
        # Forward: multi-positive soft contrastive loss (mol→morph)        
        # Compute similarity matrices
        logits_YY = torch.matmul(emb_b, emb_b.T)  # [n_valid, n_valid] - morphological self-similarity
        
        # Compute soft targets from momentum encoder (morphological self-similarity)
        if self.gradient_stop:
            soft_targets = F.softmax(logits_YY.detach() * scale_momentum, dim=1)  # Stop gradients on targets
        else:
            soft_targets = F.softmax(logits_YY * scale_momentum, dim=1)  # Allow gradients through targets
        
        # Multi-positive soft contrastive loss: -sum(soft_targets * log_softmax(predictions))
        log_probs = F.log_softmax(logits_XY * scale_main, dim=1)  # [n_valid, n_valid] - apply scaling inside softmax
        loss_forward = -(soft_targets * log_probs).sum(dim=1).mean()  # Multi-positive soft InfoNCE
        
        # Backward: hard contrastive loss (morph→mol)
        labels = torch.arange(logits_XY.size(0), device=logits_XY.device)
        loss_backward = F.cross_entropy(logits_XY.T, labels)
        
        # Collapse detection for morphological embeddings
        if is_training:
            # Create labels for morphological data (diagonal = positive, off-diagonal = negative)
            morph_labels = torch.eye(logits_XY.size(0), device=logits_XY.device)
            probs = F.softmax(logits_XY * scale_main, dim=1)
            check_collapse(emb_b, probs, self, "morph", eps=1e-6, labels=morph_labels)
        
        return 0.5 * (loss_forward + loss_backward), loss_forward, loss_backward

    def _compute_genomic_soft_contrastive_loss(self, logits_ac, emb_c, batch_indices, is_training=True):
        """Compute genomic contrastive loss: soft forward + hard backward."""
        if logits_ac is None or logits_ac.size(0) == 0:
            zero_loss = torch.tensor(0.0, device=logits_ac.device if logits_ac is not None else torch.device('cpu'))
            return zero_loss, zero_loss, zero_loss
    
        # Forward: multi-positive soft contrastive loss (mol→genomic)
        # Apply temperature scaling explicitly
        scale_main = self.temperature_main if is_training else 1.0
        scale_momentum = self.temperature_momentum_genomic if is_training else 1.0
        
        # Compute similarity matrices
        logits_YY = torch.matmul(emb_c, emb_c.T)  # [total_conditions, total_conditions] - genomic self-similarity
        
        # Compute soft targets from momentum encoder (genomic self-similarity)
        if self.gradient_stop:
            soft_targets = F.softmax(logits_YY.detach() * scale_momentum, dim=1)  # Stop gradients on targets
        else:
            soft_targets = F.softmax(logits_YY * scale_momentum, dim=1)  # Allow gradients through targets
        
        # Multi-positive soft contrastive loss: -sum(soft_targets * log_softmax(predictions))
        log_probs = F.log_softmax(logits_ac * scale_main, dim=1)  # [total_conditions, total_conditions] - apply scaling inside softmax
        loss_forward = -(soft_targets * log_probs).sum(dim=1).mean()  #Multi-positive soft InfoNCE
        
        # Backward: hard contrastive loss using multi-positive InfoNCE (genomic→mol)
        mol_ids = batch_indices  # [total_conditions] - which molecule each condition belongs to
        labels = (mol_ids[:, None] == mol_ids[None, :]).float()  # [n_conditions, n_conditions]
        
        probs_bwd = F.softmax(logits_ac.T * scale_main, dim=1)  # [n_conditions, n_conditions]
        pos_probs_bwd = (probs_bwd * labels.T).sum(dim=1)  # [n_conditions] - sum of positive probabilities per row
        loss_backward = -torch.log(pos_probs_bwd + 1e-8).mean()  # Multi-positive InfoNCE
        
        # Collapse detection for genomic embeddings
        if is_training:
            probs = F.softmax(logits_ac * scale_main, dim=1)
            check_collapse(emb_c, probs, self, "genomic", eps=1e-6, labels=labels)
        
        return 0.5 * (loss_forward + loss_backward), loss_forward, loss_backward

    def _compute_contrastive_losses(self, logits_ab, logits_ac, emb_a, emb_b, emb_c, inputs, step_name):
        """Compute bidirectional contrastive losses with soft/hard morphological losses."""
        loss = 0
        n_valid_losses = 0
        logs = {}
                
        # 1. Morphological contrastive loss (soft forward + hard backward)
        if logits_ab is not None and emb_b is not None:
            b_valid = ~torch.all(inputs['x_b'] == -1, dim=1)
            if b_valid.any():
                morphological_loss, loss_forward, loss_backward = self._compute_morphological_contrastive_loss(
                    logits_ab[b_valid], emb_b, step_name == "train"
                )
                
                loss += morphological_loss
                n_valid_losses += 1
                logs.update({
                    "morphological_loss": float(morphological_loss.cpu().detach().item()),
                    "loss_ab": float(loss_forward.cpu().detach().item()),
                    "loss_ba": float(loss_backward.cpu().detach().item()),
                })
        
        # 2. Genomic contrastive loss (soft forward + hard backward)
        if logits_ac is not None and logits_ac.size(0) > 0 and emb_c is not None and emb_a is not None:
            # Get batch indices for genomic soft contrastive loss
            batch_indices = inputs['batch_indices']
            
            genomic_loss, loss_forward, loss_backward = self._compute_genomic_soft_contrastive_loss(
                logits_ac, emb_c, batch_indices, step_name == "train"
            )
            
            loss += genomic_loss
            n_valid_losses += 1
            
            # Get molecular grouping info for logging
            unique_batch_indices, inverse_indices = torch.unique(batch_indices, return_inverse=True)
            n_molecules_with_genomic = len(unique_batch_indices)
            n_conditions = logits_ac.size(0)
            
            logs.update({
                "genomic_loss": float(genomic_loss.cpu().detach().item()),
                "loss_ac": float(loss_forward.cpu().detach().item()), 
                "loss_ca": float(loss_backward.cpu().detach().item()),
                "n_genomic_conditions": float(n_conditions),
                "n_molecules_with_genomic": float(n_molecules_with_genomic),
                "avg_conditions_per_molecule": float(n_conditions / max(n_molecules_with_genomic, 1)),
            })
        else:
            # When no genomic data is available, only include the count metrics
            logs.update({
                "n_genomic_conditions": 0.0,
                "n_molecules_with_genomic": 0.0,
                "avg_conditions_per_molecule": 0.0
            })
        
        # Average loss over number of modalities
        loss = loss / max(n_valid_losses, 1)
        logs["n_valid_losses"] = float(n_valid_losses)
        
        return loss, logs

