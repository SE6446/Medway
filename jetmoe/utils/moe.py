import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import scattermoe
    _scattermoe_available = True
except:
    print('Scattermoe is not available, using pytorch implementation.')
    from .parallel_experts import ParallelExperts, compute_gating
    _scattermoe_available = False

from .gate import top_k_gating


class MoE(nn.Module):
    """
    A Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    

    Args:
        input_size: integer - size of the input
        head_size: integer - size of the expert's hidden layer
        num_experts: an integer - number of experts
        top_k: an integer - how many experts to use for each batch element
        bias: a boolean - whether to include bias in linear layers
        activation: an activation function to apply to expert's outputs
        acc_aux_loss: a boolean - whether to accumulate auxiliary loss
        hidden_size: an integer - hidden size of the experts
        gating_dropout: a float - dropout rate for gating network
        sample_topk: an integer - how many experts to sample during training
        gating_size: an integer - size of the gating network
        aux_loss: a string - type of auxiliary loss ('mi' or 'sparse')
        gate_type: a string - type of gating mechanism ('mlp' or 'topk')
    """

    def __init__(
        self, 
        input_size, 
        hidden_size, 
        num_experts, 
        top_k,
        bias=True, 
        activation=None, 
        glu=True,
        ):
        super(MoE, self).__init__()

        self.num_experts = num_experts
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.glu = glu
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(input_size))
            torch.nn.init.zeros_(self.bias)
        else:
            self.bias = None

        if _scattermoe_available:
            self.input_linear = scattermoe.parallel_experts.ParallelExperts(num_experts, input_size, hidden_size * 2 if glu else hidden_size)
            self.output_linear = scattermoe.parallel_experts.ParallelExperts(num_experts, hidden_size, input_size)
        else:
            self.input_linear = ParallelExperts(num_experts, input_size, hidden_size * 2 if glu else hidden_size, bias=False)
            self.output_linear = ParallelExperts(num_experts, hidden_size, input_size, bias=False)

        self.top_k = min(top_k, self.num_experts)
        self.activation = activation

        self.router = top_k_gating(
            input_size=input_size, 
            num_experts=num_experts, 
            top_k=top_k, 
            )

    def extra_repr(self):
        return 'k={}, e={}'.format(
            self.top_k, self.num_experts)

    def get_aux_loss_and_clear(self):
        """
        Get the accumulated auxiliary loss and clear it.

        Returns:
            float: Accumulated auxiliary loss.
        """

        return self.gate.get_aux_loss_and_clear()
    
    def compute_gate(self, x):
        top_k_indices, self.top_k_gates = self.router(x)

        if _scattermoe_available:
            with torch.no_grad():
                self.sorted_expert_idxs, self.sorted_scattered_idxs = scattermoe.kernels.ops.flatten_and_sort(top_k_indices)
                self.padded_block_idxs, self.expert_offsets = scattermoe.kernels.ops.padded_block_indices(self.sorted_expert_idxs, self.num_experts)
        else:
            self.batch_gates, self.batch_index, expert_size, self.index_sorted_experts =\
                compute_gating(self.top_k, self.num_experts, self.top_k_gates, top_k_indices)
            self.expert_size = expert_size.tolist()

        return self.router.loss

    def scattermoe_forward(self, x):
        """
        Forward pass of the mixture of experts layer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)

        loss = self.compute_gate(x)

        h = self.input_linear(
            x, self.top_k,
            self.sorted_expert_idxs, self.sorted_scattered_idxs,
            self.padded_block_idxs, self.expert_offsets,
            grouped_out=True
        )

        if self.glu:
            h, g = h.chunk(2, dim=-1)
            h = self.activation(h) * g
        else:
            h = self.activation(h)

        y = self.output_linear(
            h, 1, 
            self.sorted_expert_idxs, self.sorted_scattered_idxs,
            self.padded_block_idxs, self.expert_offsets,
            grouped_in=True,
            gates=self.top_k_gates,
        )

        y = y.view(bsz, length, self.input_size)
        if self.bias is not None:
            y = y + self.bias
        return y, loss
    
    def torch_forward(self, x):
        """
        Forward pass of the mixture of experts layer.

        Args:
            x (Tensor): Input tensor.
            skip_mask (Tensor): Skip mask tensor.
            sample_topk (int): Number of experts to sample during training.
            multiply_by_gates (bool): Whether to multiply outputs by gating values.

        Returns:
            Tensor: Output tensor.
            float: Gating loss.
        """
        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)
        loss = self.compute_gate(x)

        expert_inputs = x[self.batch_index]
        h = self.input_linear(expert_inputs, self.expert_size)
        if self.glu:
            h, g = h.chunk(2, dim=-1)
            h = self.activation(h) * g
        else:
            h = self.activation(h)
        expert_outputs = self.output_linear(h, self.expert_size)

        expert_outputs = expert_outputs * self.batch_gates[:, None]

        zeros = torch.zeros(
            (bsz * length, self.input_size),
            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.batch_index, expert_outputs)
        y = y.view(bsz, length, self.input_size)
        if self.bias is not None:
            y = y + self.bias
        return y, loss
    
    def single_forward(self, x):
        bsz, length, emb_size = x.size()

        x = x.reshape(1, self.input_size)
        top_k_indices, top_k_gates = self.router(x)
        loss = self.router.loss

        y_list = []
        for i in range(self.top_k):
            expert_idx = top_k_indices[0,i]

            h = F.linear(x, self.input_linear.weight[expert_idx])
            if self.glu:
                h, g = h.chunk(2, dim=-1)
                h = self.activation(h) * g
            else:
                h = self.activation(h)
            y = F.linear(h, self.output_linear.weight[expert_idx]) * top_k_gates[0,i]

            y_list.append(y)
        
        y = sum(y_list)
        y = y.view(bsz, length, self.input_size)
        if self.bias is not None:
            y = y + self.bias
        return y, loss
    
    def forward(self, x):
        """
        Forward pass of the mixture of experts layer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        bsz, length, emb_size = x.size()
        if bsz * length ==1:
            return self.single_forward(x)
        elif _scattermoe_available:
            return self.scattermoe_forward(x)
        else:
            return self.torch_forward(x)

    def scattermoe_map(self, x):
        """
        Map input through the mixture of experts layer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)
        loss = self.compute_gate(x)

        y = self.input_linear(
            x, self.top_k,
            self.sorted_expert_idxs, self.sorted_scattered_idxs,
            self.padded_block_idxs, self.expert_offsets,
        )
        y = y.view(bsz, length, self.top_k, -1)
        return y, loss
    
    def single_map(self, x):
        bsz, length, emb_size = x.size()

        x = x.reshape(1, self.input_size)
        self.top_k_indices, self.top_k_gates = self.router(x)
        loss = self.router.loss

        y_list = []
        for i in range(self.top_k):
            expert_idx = self.top_k_indices[0,i]
            y = F.linear(x, self.input_linear.weight[expert_idx])
            y_list.append(y)
        y = torch.cat(y_list, dim=0)
        y = y.view(bsz, length, self.top_k, -1)
        return y, loss
    
    def torch_map(self, x):
        """
        
        Args:
            x: tensor shape [batch_size, input_size]
            train: a boolean scalar.
            loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
            y: a tensor with shape [batch_size, output_size].
            extra_training_loss: a scalar.  This should be added into the overall
            training loss of the model.  The backpropagation of this loss
            encourages all experts to be approximately equally used across a batch.
        """
        """
        Map input through the mixture of experts layer.

        Args:
            x (Tensor): Input tensor.
            skip_mask (Tensor): Skip mask tensor.
            sample_topk (int): Number of experts to sample during training.
            return_indices (bool): Whether to return expert indices.

        Returns:
            Tensor: Output tensor.
            float: Gating loss.
        """
        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)
        loss = self.compute_gate(x)

        expert_inputs = x[self.batch_index]
        expert_outputs = self.input_linear(expert_inputs, self.expert_size)

        zeros = torch.zeros((bsz * length * self.top_k, self.hidden_size), 
            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.index_sorted_experts, expert_outputs)
        y = y.view(bsz, length, self.top_k, -1)
        return y, loss
    
    def map(self, x):
        """
        Map input through the mixture of experts layer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        bsz, length, emb_size = x.size()
        if bsz * length ==1:
            return self.single_map(x)
        elif _scattermoe_available:
            return self.scattermoe_map(x)
        else:
            return self.torch_map(x)

    def scattermoe_reduce(self, x):
        """
        Reduce the mapped output.

        Args:
            x (Tensor): Mapped output tensor.

        Returns:
            Tensor: Reduced output tensor.
        """
        
        bsz, length, k, emb_size = x.size()
        assert k == self.top_k
        x = x.reshape(-1, emb_size)

        y = self.output_linear(
            x, 1, 
            self.sorted_expert_idxs, self.sorted_scattered_idxs,
            self.padded_block_idxs, self.expert_offsets,
            gates=self.top_k_gates,
        )
        y = y.view(bsz, length, self.input_size)
        return y
    
    def single_reduce(self, x):
        bsz, length, k, emb_size = x.size()

        x = x.reshape(k, emb_size)

        y_list = []
        for i in range(self.top_k):
            expert_idx = self.top_k_indices[0,i]
            y = F.linear(x[i], self.output_linear.weight[expert_idx]) * self.top_k_gates[0,i]
            y_list.append(y)
        y = sum(y_list)
        y = y.view(bsz, length, self.input_size)
        return y

    def torch_reduce(self, x):
        """
        Reduce the mapped output.

        Args:
            x (Tensor): Mapped output tensor.
            multiply_by_gates (bool): Whether to multiply outputs by gating values.

        Returns:
            Tensor: Reduced output tensor.
        """
        
        bsz, length, k, emb_size = x.size()
        x = x.reshape(-1, emb_size)

        expert_inputs = x[self.index_sorted_experts]
        expert_outputs = self.output_linear(expert_inputs, self.expert_size)

        expert_outputs = expert_outputs * self.batch_gates[:, None]

        zeros = torch.zeros((bsz * length, self.input_size), 
            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.batch_index, expert_outputs)
        y = y.view(bsz, length, self.input_size)
        return y
    
    def reduce(self, x):
        """
        Reduce the mapped output.

        Args:
            x (Tensor): Mapped output tensor.

        Returns:
            Tensor: Reduced output tensor.
        """
        bsz, length, k, emb_size = x.size()
        if bsz * length ==1:
            return self.single_reduce(x)
        elif _scattermoe_available:
            return self.scattermoe_reduce(x)
        else:
            return self.torch_reduce(x)