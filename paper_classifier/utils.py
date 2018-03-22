import torch
from torch.autograd import Variable


def sort_batch_by_length(tensor: torch.autograd.Variable,
                         sequence_lengths: torch.autograd.Variable):
    if not isinstance(tensor, Variable) or not isinstance(sequence_lengths, Variable):
        raise ValueError("Both the tensor and sequence lengths must "
                         "be torch.autograd.Variables.")

    sorted_sequence_lengths, permutation_index = sequence_lengths.sort(0, descending=True)
    sorted_tensor = tensor.index_select(0, permutation_index)

    # This is ugly, but required - we are creating a new variable at runtime, so we
    # must ensure it has the correct CUDA vs non-CUDA type. We do this by cloning and
    # refilling one of the inputs to the function.
    index_range = sequence_lengths.data.clone().copy_(
        torch.arange(0, len(sequence_lengths)))
    # This is the equivalent of zipping with index, sorting by the original
    # sequence lengths and returning the now sorted indices.
    index_range = Variable(index_range.long())
    _, reverse_mapping = permutation_index.sort(0, descending=False)
    restoration_indices = index_range.index_select(0, reverse_mapping)
    return sorted_tensor, sorted_sequence_lengths, restoration_indices, permutation_index


def extract_final_output(output, lengths, batch_first=True):
    idx = (lengths - 1).view(-1, 1).expand(
        lengths.size(0), output.size(2))
    time_dimension = 1 if batch_first else 0
    idx = idx.unsqueeze(time_dimension)
    if output.is_cuda:
        idx = idx.cuda(output.data.get_device())
    # Shape: (batch_size, hidden_size)
    last_output = output.gather(
        time_dimension, Variable(idx)).squeeze(time_dimension)
    return last_output
