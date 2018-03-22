import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from .utils import extract_final_output, sort_batch_by_length


class AcademicPaperClassifier(nn.Module):
    def __init__(self, embedding_matrix, title_encoder_hidden_size,
                 title_encoder_num_layers, title_encoder_dropout,
                 abstract_encoder_hidden_size,
                 abstract_encoder_num_layers,
                 abstract_encoder_dropout,
                 feedforward_hidden_dims,
                 feedforward_activations,
                 feedforward_dropouts):
        super(AcademicPaperClassifier, self).__init__()

        vocab_size = embedding_matrix.size(0)
        embedding_dim = embedding_matrix.size(1)
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight = nn.Parameter(embedding_matrix)

        self.title_encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=title_encoder_hidden_size,
            num_layers=title_encoder_num_layers,
            dropout=title_encoder_dropout,
            bidirectional=True, batch_first=True)
        self.abstract_encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=abstract_encoder_hidden_size,
            num_layers=abstract_encoder_num_layers,
            dropout=abstract_encoder_dropout,
            bidirectional=True, batch_first=True)

        # Build the feedforward classifier
        assert (len(feedforward_hidden_dims) ==
                len(feedforward_activations) ==
                len(feedforward_dropouts))
        classifier_input_size = ((title_encoder_hidden_size * 2) +
                                 (abstract_encoder_hidden_size * 2))
        classifier_feedforward_components = []
        for hidden, activation, dropout in zip(feedforward_hidden_dims,
                                               feedforward_activations,
                                               feedforward_dropouts):
            classifier_feedforward_components.append(
                nn.Linear(classifier_input_size, hidden))
            classifier_input_size = hidden
            classifier_feedforward_components.append(
                ACTIVATIONS[activation]())
            classifier_feedforward_components.append(
                nn.Dropout(dropout))
        # Add output linear projection to 3 labels
        classifier_feedforward_components.append(
            nn.Linear(classifier_input_size, 3))
        self.classifier_feedforward = nn.Sequential(
            *classifier_feedforward_components)
        self.global_step = 0

    def forward(self, titles, title_lengths, abstracts, abstract_lengths):
        embedded_titles = self.embedding(titles)
        embedded_abstracts = self.embedding(abstracts)

        (sorted_titles, sorted_title_lengths,
         titles_unsort_indices, _) = sort_batch_by_length(
             embedded_titles, title_lengths)
        (sorted_abstracts, sorted_abstract_lengths,
         abstracts_unsort_indices, _) = sort_batch_by_length(
             embedded_abstracts, abstract_lengths)

        # Encode the titles.
        packed_titles = pack_padded_sequence(
            sorted_titles, sorted_title_lengths.data.tolist(),
            batch_first=True)
        sorted_titles_hiddens, _ = self.title_encoder(packed_titles)
        sorted_titles_hiddens, _ = pad_packed_sequence(
            sorted_titles_hiddens, batch_first=True)
        titles_hiddens = sorted_titles_hiddens[titles_unsort_indices]
        encoded_titles = extract_final_output(titles_hiddens, title_lengths.data)

        # Encode the abstracts.
        packed_abstracts = pack_padded_sequence(
            sorted_abstracts, sorted_abstract_lengths.data.tolist(),
            batch_first=True)
        sorted_abstracts_hiddens, _ = self.abstract_encoder(packed_abstracts)
        sorted_abstracts_hiddens, _ = pad_packed_sequence(
            sorted_abstracts_hiddens, batch_first=True)
        abstracts_hiddens = sorted_abstracts_hiddens[abstracts_unsort_indices]
        encoded_abstracts = extract_final_output(abstracts_hiddens, abstract_lengths.data)

        logits = self.classifier_feedforward(
            torch.cat([encoded_titles, encoded_abstracts], dim=-1))
        return logits


ACTIVATIONS = {
    "linear": lambda: lambda x: x,
    "relu": torch.nn.ReLU,
    "relu6": torch.nn.ReLU6,
    "elu": torch.nn.ELU,
    "prelu": torch.nn.PReLU,
    "leaky_relu": torch.nn.LeakyReLU,
    "threshold": torch.nn.Threshold,
    "hardtanh": torch.nn.Hardtanh,
    "sigmoid": torch.nn.Sigmoid,
    "tanh": torch.nn.Tanh,
    "log_sigmoid": torch.nn.LogSigmoid,
    "softplus": torch.nn.Softplus,
    "softshrink": torch.nn.Softshrink,
    "softsign": torch.nn.Softsign,
    "tanhshrink": torch.nn.Tanhshrink,
}
