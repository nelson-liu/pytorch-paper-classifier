import os
from unittest import TestCase

from paper_classifier.data import read_data, VENUE2IDX


class TestReadData(TestCase):
    project_root = os.path.abspath(os.path.realpath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)), os.pardir)))

    def test_read_from_file(self):
        train_dataset, train_vocab = read_data(
            os.path.join(self.project_root, "tests", "fixtures", "s2_papers.jsonl"),
            vocabulary=None)
        idx2word = {v: k for k, v in train_vocab.items()}

        true_instances = [
            {
                "title": ["Interferring", "Discourse", "Relations", "in", "Context"],
                "abstract": ["We", "investigate", "various", "contextual", "effects"],
                "venue": "ACL"
            },
            {
                "title": ["GRASPER", ":", "A", "Permissive", "Planning", "Robot"],
                "abstract": ["Execut", "ion", "of", "classical", "plans"],
                "venue": "AI"
            },
            {
                "title": ["Route", "Planning", "under", "Uncertainty", ":",
                          "The", "Canadian", "Traveller", "Problem"],
                "abstract": ["The", "Canadian", "Traveller", "problem", "is"],
                "venue": "AI"
            }]

        def check_read_instance(read_instance, true_instance):
            assert [idx2word[t] for t in read_instance[0]] == true_instance["title"]
            assert read_instance[1] == len(true_instance["title"])
            assert [idx2word[t] for t in
                    read_instance[2][:5]] == true_instance["abstract"]
            assert read_instance[4] == VENUE2IDX[true_instance["venue"]]

        assert len(train_dataset) == 10
        for idx, true_instance in enumerate(true_instances):
            check_read_instance(train_dataset[idx], true_instance)
