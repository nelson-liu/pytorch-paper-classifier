import os
from unittest import TestCase

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy

from paper_classifier.data import read_data
from paper_classifier.data import PaperDataset
from paper_classifier.academic_paper_classifier import AcademicPaperClassifier


class TestAcademicPaperClassifier(TestCase):
    project_root = os.path.abspath(os.path.realpath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)), os.pardir)))

    def test_training(self):
        batch_size = 3
        train_dataset, train_vocab = read_data(
            os.path.join(self.project_root, "tests",
                         "fixtures", "s2_papers.jsonl"),
            vocabulary=None)
        train_dataloader = DataLoader(
            dataset=train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=4,
            collate_fn=PaperDataset.collate_fn)
        embedding_dim = 50
        embedding_matrix = torch.FloatTensor(
            len(train_vocab), embedding_dim).normal_()
        model = AcademicPaperClassifier(
            embedding_matrix=embedding_matrix,
            title_encoder_hidden_size=15,
            title_encoder_num_layers=2,
            title_encoder_dropout=0.1,
            abstract_encoder_hidden_size=20,
            abstract_encoder_num_layers=2,
            abstract_encoder_dropout=0.1,
            feedforward_hidden_dims=[50, 20],
            feedforward_activations=["relu", "tanh"],
            feedforward_dropouts=[0.1, 0.1])
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()))
        for batch in train_dataloader:
            title, title_lengths, abstract, abstract_lengths, venue = batch
            title = Variable(title)
            title_lengths = Variable(title_lengths)
            abstract = Variable(abstract)
            abstract_lengths = Variable(abstract_lengths)
            venue = Variable(venue)

            logits = model(title, title_lengths, abstract, abstract_lengths)
            batch_loss = cross_entropy(logits, venue)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
