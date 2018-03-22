import gzip
import itertools
import json
import logging
import mmap

import numpy as np
import spacy
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)
spacy_en = spacy.load("en")
VENUE2IDX = {"AI": 0, "ML": 1, "ACL": 2}


def load_embeddings(glove_path, vocab):
    """
    Create an embedding matrix for a Vocabulary.
    """
    vocab_size = len(vocab)
    words_to_keep = set(vocab.keys())
    idx2words = {v: k for k, v in vocab.items()}

    glove_embeddings = {}
    embedding_dim = None

    logger.info("Reading GloVe embeddings from {}".format(glove_path))
    with gzip.open(glove_path, "rb") as glove_file:
        for line in glove_file:
            fields = line.decode("utf-8").strip().split(" ")
            word = fields[0]
            if word in words_to_keep:
                vector = np.asarray(fields[1:], dtype="float32")
                if embedding_dim is None:
                    embedding_dim = len(vector)
                else:
                    assert embedding_dim == len(vector)
                glove_embeddings[word] = vector

    all_embeddings = np.asarray(list(glove_embeddings.values()))
    embeddings_mean = float(np.mean(all_embeddings))
    embeddings_std = float(np.std(all_embeddings))
    logger.info("Initializing {}-dimensional pretrained "
                "embeddings for {} tokens".format(
                    embedding_dim, vocab_size))
    embedding_matrix = torch.FloatTensor(
        vocab_size, embedding_dim).normal_(
            embeddings_mean, embeddings_std)
    # Manually zero out the embedding of the padding token (0).
    embedding_matrix[0].fill_(0)
    # This starts from 1 because 0 is the padding token, which
    # we don't want to modify.
    for i in range(1, vocab_size):
        word = idx2words[i]

        # If we don't have a pre-trained vector for this word,
        # we don't change the row and the word has random initialization.
        if word in glove_embeddings:
            embedding_matrix[i] = torch.FloatTensor(
                glove_embeddings[word])
    return embedding_matrix


def read_data(data_path, vocabulary=None, has_labels=True):
    """
    Read a data file, and numericalize it if a
    vocabulary object is provided. Else, create a
    vocabulary object.
    """
    numericalized_titles = []
    numericalized_abstracts = []
    venues = []

    logger.info("Reading instances from {}".format(data_path))

    index = False
    if vocabulary is None:
        index = True
        vocabulary = {"<PAD>": 0, "<UNK>": 1}

    with open(data_path) as data_file:
        for line in tqdm(data_file, total=get_num_lines(data_path)):
            line = line.strip("\n")
            if not line:
                continue
            paper_json = json.loads(line)
            title = paper_json['title']
            abstract = paper_json['paperAbstract']
            tokenized_title = [
                tok.text for tok in spacy_en.tokenizer(title)]
            tokenized_abstract = [
                tok.text for tok in spacy_en.tokenizer(abstract)]
            # Index before numericalizing, if applicable.
            if index:
                for word in itertools.chain(tokenized_abstract,
                                            tokenized_title):
                    if word not in vocabulary:
                        vocabulary[word] = len(vocabulary)

            # Numericalize the title and abstract text.
            numericalized_titles.append(
                [vocabulary.get(word, 1) for word in tokenized_title])
            numericalized_abstracts.append(
                [vocabulary.get(word, 1) for word in tokenized_abstract])
            if has_labels:
                venue = paper_json['venue']
                # Numericalize the venue.
                venues.append(VENUE2IDX[venue])
    return (PaperDataset(numericalized_titles,
                         numericalized_abstracts,
                         venues if has_labels else None),
            vocabulary)


class PaperDataset(Dataset):
    def __init__(self, numericalized_titles, numericalized_abstracts,
                 numericalized_venues=None):
        self.titles = numericalized_titles
        self.abstracts = numericalized_abstracts
        self.venues = numericalized_venues

    def __getitem__(self, idx):
        title = self.titles[idx]
        title_length = len(title)

        abstract = self.abstracts[idx]
        abstract_length = len(abstract)

        return (torch.LongTensor(title), title_length,
                torch.LongTensor(abstract), abstract_length,
                None if self.venues is None else self.venues[idx])

    def __len__(self):
        return len(self.titles)

    @staticmethod
    def collate_fn(batch):
        """
        Given a list of examples (each from __getitem__),
        combine them to form a single batch by padding.

        Returns:
        -------
        batch_padded_example_text: LongTensor
          LongTensor of shape (batch_size, longest_sequence_length) with the
          padded text for each example in the batch.
        length: LongTensor
          LongTensor of shape (batch_size,) with the unpadded length of the example.
        example_label: LongTensor
          LongTensor of shape (batch_size,) with the label of the example.
        """
        batch_padded_numericalized_titles = []
        batch_title_lengths = torch.LongTensor(
            [example[1] for example in batch])
        batch_padded_numericalized_abstracts = []
        batch_abstract_lengths = torch.LongTensor(
            [example[3] for example in batch])
        has_labels = (batch[0][4] is not None)
        batch_venues = (
            torch.LongTensor([example[4] for example in batch]) if
            has_labels else None)

        # Get the length of the longest title in the batch
        max_title_length = max(
            batch, key=lambda example: example[1])[1]

        # Get the length of the longest abstract in the batch
        max_abstract_length = max(
            batch, key=lambda example: example[3])[3]

        # Iterate over each example in the batch
        for example in batch:
            # Unpack the example (returned from __getitem__)
            title, title_length, abstract, abstract_length = example[:4]

            title_padding_length = max_title_length - title_length
            title_padding = torch.zeros(title_padding_length).long()
            padded_title = torch.cat((title, title_padding))

            abstract_padding_length = max_abstract_length - abstract_length
            abstract_padding = torch.zeros(abstract_padding_length).long()
            padded_abstract = torch.cat((abstract, abstract_padding))

            # Add the padded example to our batch
            batch_padded_numericalized_titles.append(padded_title)
            batch_padded_numericalized_abstracts.append(padded_abstract)

        # Stack the list of LongTensors to make the batch.
        return (torch.stack(batch_padded_numericalized_titles),
                batch_title_lengths,
                torch.stack(batch_padded_numericalized_abstracts),
                batch_abstract_lengths,
                batch_venues)


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines
