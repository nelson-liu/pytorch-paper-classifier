import argparse
import logging
import os
import shutil
import sys

from tensorboardX import SummaryWriter
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__)))
from paper_classifier.data import load_embeddings, read_data
from paper_classifier.data import PaperDataset, VENUE2IDX
from paper_classifier.academic_paper_classifier import AcademicPaperClassifier

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    project_root = os.path.abspath(os.path.realpath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)))))

    parser.add_argument("--train-path", type=str,
                        default=os.path.join(
                            project_root, "data", "train.jsonl"),
                        help="Path to the training data.")
    parser.add_argument("--dev-path", type=str,
                        default=os.path.join(
                            project_root, "data", "dev.jsonl"),
                        help="Path to the validation/development data.")
    parser.add_argument("--test-path", type=str,
                        help=("Path to test data, used to "
                              "evaluate a loaded model."))
    parser.add_argument("--glove-path", type=str,
                        default=os.path.join(
                            project_root, "data", "glove.6B.100d.txt.gz"),
                        help="Path to gzipped GloVe-formatted word vectors.")

    parser.add_argument("--title-encoder-hidden-size", type=int, default=100,
                        help="Hidden size to use in title encoder biLSTM.")
    parser.add_argument("--title-encoder-num-layers", type=int, default=1,
                        help="Number of layers to use in title encoder biLSTM.")
    parser.add_argument("--title-encoder-dropout", type=float, default=0.2,
                        help="Dropout to use in title encoder biLSTM.")

    parser.add_argument("--abstract-encoder-hidden-size", type=int, default=100,
                        help="Hidden size to use in abstract encoder biLSTM.")
    parser.add_argument("--abstract-encoder-num-layers", type=int, default=1,
                        help="Number of layers to use in abstract encoder biLSTM.")
    parser.add_argument("--abstract-encoder-dropout", type=float, default=0.2,
                        help="Dropout to use in abstract encoder biLSTM.")

    parser.add_argument("--feedforward-hidden-dims", nargs="+", default=[200],
                        type=int, help=("Hidden size(s) to use in layer(s) of "
                                        "feedforward network before output layer."))
    parser.add_argument("--feedforward-activations", nargs="+", default=["relu"],
                        type=str, help=("Activation(s) to use in layer(s) of feedforward "
                                        "network before output layer."))
    parser.add_argument("--feedforward-dropouts", nargs="+", default=[0.2],
                        type=str, help=("Dropout(s) to use in layer(s) of feedforward "
                                        "network before output layer."))

    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size to use in training and evaluation.")
    parser.add_argument("--num-epochs", type=int, default=25,
                        help="Number of epochs to train for.")

    parser.add_argument("--optimizer", type=str, default="adagrad",
                        choices=["adagrad", "adadelta", "adam", "sgd", "rmsprop"],
                        help="The optimizer to use.")
    parser.add_argument("--lr", type=float, default=0.5,
                        help="The learning rate to use.")
    parser.add_argument("--load-path", type=str,
                        help=("Path to load a saved model from and "
                              "evaluate on test data. May not be "
                              "used with --save-dir."))
    parser.add_argument("--predict", action="store_true",
                        help=("Make predictions on test data without labels (as "
                              "opposed to calculating accuracy and loss on a test "
                              "set with labels)."))
    parser.add_argument("--save-dir", type=str,
                        help=("Path to save model checkpoints and logs. "
                              "Required if not using --load-path. "
                              "May not be used with --load-path."))
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed to use")
    parser.add_argument("--cuda", action="store_true",
                        help="Train or evaluate with GPU.")
    args = parser.parse_args()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            logger.warning("\033[35mGPU available but not running with "
                           "CUDA (use --cuda to turn on.)\033[0m")
        else:
            torch.cuda.manual_seed(args.seed)

    # Load a model from checkpoint and evaluate it on test data.
    if args.load_path:
        logger.info("Loading saved model from {}".format(args.load_path))

        # If evaluating with CPU, force all tensors to CPU.
        # This lets us load models trained on the GPU on the CPU.
        saved_state_dict = torch.load(args.load_path,
                                      map_location=None if args.cuda
                                      else lambda storage, loc: storage)

        # Extract the contents of the state dictionary.
        model_weights = saved_state_dict["model_weights"]
        model_init_arguments = saved_state_dict["init_arguments"]

        # Reconstruct a model of the proper type with the init arguments.
        saved_model = AcademicPaperClassifier(**model_init_arguments)
        # Load the weights
        saved_model.load_state_dict(model_weights)

        logger.info("Successfully loaded model!")

        # Move model to GPU if using CUDA.
        if args.cuda:
            saved_model = saved_model.cuda()

        # Load the serialized train_vocab.
        vocab_path = os.path.join(os.path.dirname(args.load_path),
                                  "train_vocab")
        logger.info("Loading train vocabulary from {}".format(vocab_path))
        train_vocab = torch.load(vocab_path)
        logger.info("Successfully loaded train vocabulary!")

        # Make predictions on the test set.
        logger.info("Reading test set at {}".format(
            args.test_path))
        has_labels = not args.predict
        test_dataset, _ = read_data(
            args.test_path, vocabulary=train_vocab, has_labels=has_labels)
        logger.info("Read {} test examples".format(
            len(test_dataset.instances)))

        logger.info("Running model on the test set")
        (loss, accuracy, predictions) = evaluate(
            saved_model, test_dataset, args.batch_size,
            args.cuda, has_labels=False)
        if has_labels:
            logger.info("Done evaluating on test set!")
            logger.info("Test Loss: {:.4f}".format(loss))
            logger.info("Test Accuracy: {:.4f}".format(accuracy))
        else:
            idx2venues = {v: k for k, v in VENUE2IDX.items()}
            # Output model predictions on an unlabeled test set.
            with open("test_predictions.txt", "w") as predictions_file:
                for prediction in predictions:
                    predictions_file.write("{}\n".format(idx2venues[prediction]))
            logger.info("Wrote predictions to {}".format("test_predictions.txt"))
        sys.exit(0)

    if not args.save_dir:
        raise ValueError("Must provide a value for --save-dir if training.")
    try:
        if os.path.exists(args.save_dir):
            # Save directory already exists, confirm overwrite.
            input("Save directory {} already exists. Press <Enter> "
                  "to overwrite and continue, or <Ctrl-c> to abort.".format(
                      args.save_dir))
            shutil.rmtree(args.save_dir)
        os.makedirs(args.save_dir)
    except KeyboardInterrupt:
        print()
        sys.exit(0)

    # Write Tensorboard logs to args.save_dir/logs.
    log_dir = os.path.join(args.save_dir, "logs")
    os.makedirs(log_dir)

    # Read the training and validation dataset into a
    # PaperDataset, and get a vocabulary from the train set.
    train_dataset, train_vocab = read_data(
        args.train_path, vocabulary=None)
    validation_dataset, _ = read_data(
        args.dev_path, vocabulary=train_vocab)

    # Save the training vocabulary to a file.
    vocab_path = os.path.join(args.save_dir, "train_vocab")
    logger.info("Saving train vocabulary to {}".format(vocab_path))
    torch.save(train_vocab, vocab_path)

    # Read GloVe embeddings.
    embedding_matrix = load_embeddings(args.glove_path, train_vocab)

    # Create the model
    model = AcademicPaperClassifier(
        embedding_matrix=embedding_matrix,
        title_encoder_hidden_size=args.title_encoder_hidden_size,
        title_encoder_num_layers=args.title_encoder_num_layers,
        title_encoder_dropout=args.title_encoder_dropout,
        abstract_encoder_hidden_size=args.abstract_encoder_hidden_size,
        abstract_encoder_num_layers=args.abstract_encoder_num_layers,
        abstract_encoder_dropout=args.abstract_encoder_dropout,
        feedforward_hidden_dims=args.feedforward_hidden_dims,
        feedforward_activations=args.feedforward_activations,
        feedforward_dropouts=args.feedforward_dropouts)

    logger.info(model)

    # Move model to GPU if running with CUDA.
    if args.cuda:
        model = model.cuda()
    # Create the optimizer, and only update parameters where requires_grad=True
    optimizer = OPTIMIZERS[args.optimizer](
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr)
    # Train for the specified number of epochs.
    for i in tqdm(range(args.num_epochs), unit="epoch"):
        train_epoch(model, train_dataset, validation_dataset,
                    args.batch_size, optimizer, args.save_dir,
                    log_dir, args.cuda)


def train_epoch(model, train_dataset, validation_dataset,
                batch_size, optimizer, save_dir, log_dir, cuda):
    """
    Train the model for one epoch.
    """
    # Set model to train mode (turns on dropout and such).
    model.train()
    # Create Tensorboard logger.
    writer = SummaryWriter(log_dir)

    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, collate_fn=PaperDataset.collate_fn, pin_memory=cuda)

    log_period_total_loss = 0
    log_period_num_correct = 0
    log_period_num_examples = 0

    for batch in tqdm(train_dataloader):
        title, title_lengths, abstract, abstract_lengths, venue = batch
        title, title_lengths = Variable(title), Variable(title_lengths)
        abstract, abstract_lengths = Variable(abstract), Variable(abstract_lengths)
        venue = Variable(venue)

        if cuda:
            title, title_lengths = title.cuda(), title_lengths.cuda()
            abstract, abstract_lengths = abstract.cuda(), abstract_lengths.cuda()
            venue = venue.cuda()

        # Run data through model to get logits.
        logits = model(title, title_lengths, abstract, abstract_lengths)

        # Calculate batch loss
        batch_loss = cross_entropy(logits, venue)
        log_period_total_loss += batch_loss.data[0] * len(batch[1])

        # Calculate the number of correctly classified batch items
        _, batch_predicted_labels = torch.max(logits.data, 1)
        log_period_num_correct += torch.sum(batch_predicted_labels ==
                                            venue.data)
        log_period_num_examples += venue.size(0)

        # Backprop and take a gradient step.
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        model.global_step += 1

        # Log train metrics to Tensorboard every 50 steps.
        if model.global_step % 50 == 0:
            # Calculate metrics on train set.
            loss = log_period_total_loss / log_period_num_examples
            accuracy = log_period_num_correct / log_period_num_examples

            # Log training statistics to Tensorboard
            log_to_tensorboard(writer, model.global_step, "train",
                               loss, accuracy)
            log_period_total_loss = 0
            log_period_num_correct = 0

        # Log validation metrics to Tensorboard and save model
        # every 50 steps.
        if model.global_step % 500 == 0:
            # Calculate metrics on validation set.
            (loss, accuracy, _) = evaluate(
                model, validation_dataset, batch_size,
                cuda, has_labels=True)
            # Save a checkpoint.
            save_name = ("{}_step_{}_loss_{:.3f}_"
                         "accuracy_{:.3f}.pth".format(
                             model.__class__.__name__, model.global_step,
                             loss, accuracy))
            save_model(model, save_dir, save_name)
            # Log validation statistics to Tensorboard.
            log_to_tensorboard(writer, model.global_step, "validation",
                               loss, accuracy)


def evaluate(model, evaluation_dataset, batch_size, cuda, has_labels):
    """
    Evaluate a model on an evaluation dataset.
    """
    # Set model to evaluation mode (turns off dropout and such)
    model.eval()
    predicted_labels = []

    # Build iterator
    evaluation_dataloader = DataLoader(
        dataset=evaluation_dataset, batch_size=batch_size, num_workers=4,
        collate_fn=PaperDataset.collate_fn, pin_memory=cuda)

    total_loss = 0
    total_num_correct = 0
    for batch in tqdm(evaluation_dataloader):
        title, title_lengths, abstract, abstract_lengths, venue = batch
        title, title_lengths = Variable(title), Variable(title_lengths)
        abstract, abstract_lengths = Variable(abstract), Variable(abstract_lengths)
        if has_labels:
            venue = Variable(venue)

        if cuda:
            title, title_lengths = title.cuda(), title_lengths.cuda()
            abstract, abstract_lengths = abstract.cuda(), abstract_lengths.cuda()
            if has_labels:
                venue = venue.cuda()

        # Run data through model to get logits.
        logits = model(title, title_lengths, abstract, abstract_lengths)

        # Get predicted labels
        _, batch_predicted_labels = torch.max(logits.data, 1)
        predicted_labels.append(batch_predicted_labels)

        if has_labels:
            # Calculate batch loss
            batch_loss = cross_entropy(logits, venue)
            total_loss += batch_loss.data[0] * len(batch[1])

            # Calculate the number of correctly classified batch items
            total_num_correct += torch.sum(batch_predicted_labels ==
                                           venue.data)

    # Set the model back to train mode.
    model.train()

    return (total_loss / len(evaluation_dataset) if has_labels else None,
            total_num_correct / len(evaluation_dataset) if has_labels else None,
            torch.cat(predicted_labels))


def log_to_tensorboard(writer, step, prefix, loss, accuracy):
    """
    Log metrics to Tensorboard.
    """
    writer.add_scalar("{}/loss".format(prefix), loss, step)
    writer.add_scalar("{}/accuracy".format(prefix), accuracy, step)


def save_model(model, save_dir, save_name):
    """
    Save a model to the disk.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_weights = model.state_dict()
    serialization_dictionary = {
        "model_weights": model_weights,
        "init_arguments": model.init_arguments
    }

    save_path = os.path.join(save_dir, save_name)
    torch.save(serialization_dictionary, save_path)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(levelname)s "
                        "- %(name)s - %(message)s",
                        level=logging.INFO)
    OPTIMIZERS = {
        "adam": torch.optim.Adam,
        "adagrad": torch.optim.Adagrad,
        "adadelta": torch.optim.Adadelta,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
    }
    main()
