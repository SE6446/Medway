import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.jit.script
def compute_gating(k: int, num_experts: int, top_k_gates: torch.Tensor, top_k_indices: torch.Tensor):
    """
    Compute gating values for the mixture of experts based on probabilities and top-k indices.

    Args:
        k (int): Number of experts to select.
        num_experts (int): Total number of experts.
        top_k_gates (torch.Tensor): Gating values for top-k experts (batch_size x k).
        top_k_indices (torch.Tensor): Indices of top-k experts (batch_size x k).

    Returns:
        torch.Tensor: Batch-level gating values.
        torch.Tensor: Batch-level expert indices.
        torch.Tensor: Expert size for each expert.
        torch.Tensor: Sorted indices of top-k experts.
    """
    zeros = torch.zeros([top_k_gates.size(0), num_experts], dtype=top_k_gates.dtype, device=top_k_gates.device)
    gates = zeros.scatter(1, top_k_indices, 1)
    expert_size = gates.long().sum(0)
    top_k_gates = top_k_gates.flatten()
    top_k_experts = top_k_indices.flatten()
    _, index_sorted_experts = top_k_experts.sort(0)
    batch_index = index_sorted_experts.div(k, rounding_mode="trunc")
    batch_gates = top_k_gates[index_sorted_experts]
    return batch_gates, batch_index, expert_size, index_sorted_experts


class ParallelExperts1_58b(nn.Module):
    def __init__(self, num_experts, input_size, output_size) -> None:
        """
        Initialize the ParallelExperts1_58b module.

        Args:
            num_experts (int): Number of experts.
            input_size (int): Size of the input.
            output_size (int): Size of the output.
            bias (bool): Whether to include bias terms.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, output_size, input_size))
        self.reset_parameters()
        self.num_experts = num_experts
        self.input_size = input_size
        self.output_size = output_size

    def extra_repr(self):
        return "num_experts={}, input_size={}, output_size={}".format(
            self.num_experts, self.input_size, self.output_size
        )

    def reset_parameters(self) -> None:
        """
        Reset the parameters of the model.
        """
        nn.init.uniform_(self.weight, -1.0 / self.weight.size(1), 1.0 / self.weight.size(1))

    #TODO make inference version.
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

    def forward(self, inputs, expert_size):
        """
        Forward pass of the ParallelExperts1_58b module.

        Args:
            inputs (Tensor): Input tensor.
            expert_size: Expert size information.

        Returns:
            Tensor: Output tensor.
        """
        #! Input should be pre_quantazised.
        w = self.weight
        w_quant = w + (self.__weight_quant(w) - w).detach()
        input_list = inputs.split(expert_size, dim=0)
        output_list = []
        for i in range(self.num_experts):
            output_list.append(F.linear(input_list[i], w_quant[i]))
        results = torch.cat(output_list, dim=0)
        return results