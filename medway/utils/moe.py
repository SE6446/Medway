from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .parallel_experts import ParallelExperts1_58b, compute_gating

from .gate import top_k_gating


class MoE(nn.Module):
    """
    A Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.

    Args:
        input_size: integer - size of the input
        hidden_size: integer - size of the expert's hidden layer
        num_experts: an integer - number of experts
        top_k: an integer - how many experts to use for each batch element
        bias: a boolean - whether to include bias in linear layers
        activation: an activation function to apply to expert's outputs
        glu: an boolean - whether to use GLU activation
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_experts,
        top_k,
        bias=True,
        activation=None,
        glu=True
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

        self.input_linear = ParallelExperts1_58b(num_experts, input_size, hidden_size * 2 if glu else hidden_size)
        self.output_linear = ParallelExperts1_58b(num_experts, hidden_size, input_size)

        self.top_k = min(top_k, self.num_experts)
        self.activation = activation

        self.router = top_k_gating(
            input_size=input_size,
            num_experts=num_experts,
            top_k=top_k,
        )

    def extra_repr(self):
        return "k={}, e={}".format(self.top_k, self.num_experts)

    def get_aux_loss_and_clear(self):
        """
        Get the accumulated auxiliary loss and clear it.

        Returns:
            float: Accumulated auxiliary loss.
        """

        return self.gate.get_aux_loss_and_clear()

    def compute_gate(self, x):
        top_k_indices, self.top_k_gates = self.router(x)

        self.batch_gates, self.batch_index, expert_size, self.index_sorted_experts = compute_gating(
            self.top_k, self.num_experts, self.top_k_gates, top_k_indices
        )
        self.expert_size = expert_size.tolist()

        return self.router.loss

    #TODO create an inference edition for this. This should only be used for training
    def __activation_quant(self,x):
        """ Per-token quantization to 8 bits. No grouping is needed for quantization.
        Args:
        x: an activation tensor with shape [n, d]
        Returns:
        y: a quantized activation tensor with shape [n, d]
        """
        scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        y = (x * scale).round().clamp_(-128, 127) / scale
        return y
    def __weight_quant(self,w):
        """ Per-tensor quantization to 1.58 bits. No grouping is needed for quantization.
        Args:
        w: a weight tensor with shape [d, k]
        Returns:
        u: a quantized weight with shape [d, k]
        """
        scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
        u = (w * scale).round().clamp_(-1, 1) / scale
        return u

    def batch_forward(self, x):
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
        expert_inputs = expert_inputs + (self.__activation_quant(expert_inputs) - expert_inputs).detach()
        h = self.input_linear(expert_inputs, self.expert_size)
        if self.glu:
            h, g = h.chunk(2, dim=-1)
            h = self.activation(h) * g
        else:
            h = self.activation(h)
        h_quant = h + (self.__activation_quant(h) - h).detach()
        expert_outputs = self.output_linear(h_quant, self.expert_size)

        expert_outputs = expert_outputs * self.batch_gates[:, None]

        zeros = torch.zeros((bsz * length, self.input_size), dtype=expert_outputs.dtype, device=expert_outputs.device)
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
        x_quant = x + (self.__activation_quant(x) - x).detach()

        y_list = []
        for i in range(self.top_k):
            expert_idx = top_k_indices[0, i]
            h = F.linear(x_quant, self.input_linear.weight[expert_idx] + (self.__weight_quant(self.input_linear.weight[expert_idx]) - self.input_linear.weight[expert_idx]).detach())
            if self.glu:
                h, g = h.chunk(2, dim=-1)
                h = self.activation(h) * g
            else:
                h = self.activation(h)
            # A trick for implementing Straight-Through-Estimator (STE) using detach()
            h_quant = h + (self.__activation_quant(h) - h).detach()
            y = F.linear(h_quant, self.output_linear.weight[expert_idx] + (self.__weight_quant(self.output_linear.weight[expert_idx]) - self.output_linear.weight[expert_idx]).detach()) * top_k_gates[0, i]

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
        if bsz * length == 1:
            return self.single_forward(x)
        else:
            return self.batch_forward(x)

    def single_map(self, x):
        bsz, length, emb_size = x.size()

        x = x.reshape(1, self.input_size)
        self.top_k_indices, self.top_k_gates = self.router(x)
        loss = self.router.loss
        x_quant = x + (self.__activation_quant(x) - x).detach()
        y_list = []
        for i in range(self.top_k):
            expert_idx = self.top_k_indices[0, i]
            y = F.linear(x_quant, self.input_linear.weight[expert_idx] + (self.__weight_quant(self.input_linear.weight[expert_idx]) - self.input_linear.weight[expert_idx]).detach())
            y_list.append(y)
        y = torch.cat(y_list, dim=0)
        y = y.view(bsz, length, self.top_k, -1)
        return y, loss

    def batch_map(self, x):
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

        x = x + (self.__activation_quant(x) - x).detach()
        expert_inputs = x[self.batch_index]
        expert_outputs = self.input_linear(expert_inputs, self.expert_size)

        zeros = torch.zeros(
            (bsz * length * self.top_k, self.hidden_size), dtype=expert_outputs.dtype, device=expert_outputs.device
        )
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
        if bsz * length == 1:
            return self.single_map(x)
        else:
            return self.batch_map(x)

    def single_reduce(self, x):
        bsz, length, k, emb_size = x.size()

        x = x.reshape(k, emb_size)

        y_list = []
        for i in range(self.top_k):
            expert_idx = self.top_k_indices[0, i]
            y = F.linear(x[i], self.output_linear.weight[expert_idx]) * self.top_k_gates[0, i]
            y_list.append(y)
        y = sum(y_list)
        y = y.view(bsz, length, self.input_size)
        if self.bias is not None:
            y = y + self.bias
        return y

    def batch_reduce(self, x):
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

        zeros = torch.zeros((bsz * length, self.input_size), dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.batch_index, expert_outputs)
        y = y.view(bsz, length, self.input_size)
        if self.bias is not None:
            y = y + self.bias
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
        if bsz * length == 1:
            return self.single_reduce(x)
        else:
            return self.batch_reduce(x)