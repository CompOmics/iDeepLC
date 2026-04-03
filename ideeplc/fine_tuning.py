import torch
import copy
from torch.utils.data import DataLoader
import logging
from ideeplc.predict import validate

LOGGER = logging.getLogger(__name__)


class iDeepLCFineTuner:
    """
    A class to fine-tune the iDeepLC model on a new dataset.
    """

    def __init__(
        self,
        model,
        train_data,
        loss_function,
        device="cpu",
        learning_rate=0.001,
        epochs=10,
        batch_size=256,
        validation_data=None,
        validation_split=0.1,
        patience=5,
        num_workers=0,
        pin_memory=False,
    ):
        """
        Initialize the fine-tuner with the model and data loaders.

        :param model: The iDeepLC model to be fine-tuned.
        :param train_data: Training dataset.
        :param loss_function: Loss function to use for training.
        :param device: Device to run the training on ("cpu" or "cuda").
        :param learning_rate: Learning rate for the optimizer.
        :param epochs: Number of epochs to train.
        :param batch_size: Batch size for training.
        :param validation_data: Optional validation dataset.
        :param validation_split: Fraction of training data to use for validation.
        :param patience: Number of epochs with no improvement after which training will be stopped.
        :param num_workers: Number of workers for the DataLoader.
        :param pin_memory: Whether to pin memory in the DataLoader.
        """
        self.model = model.to(device)
        self.train_data = train_data
        self.loss_function = loss_function
        self.device = device
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_data = validation_data
        self.validation_split = validation_split
        self.patience = patience
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def _freeze_layers(self, layers_to_freeze):
        """
        Freeze specified layers in the model.

        :param layers_to_freeze: List of layer names to freeze.
        """
        for name, param in self.model.named_parameters():
            if any(layer in name for layer in layers_to_freeze):
                param.requires_grad = False
                LOGGER.info(f"Freezing layer: {name}")
            else:
                param.requires_grad = True

    def prepare_data(self, data, shuffle=True):
        """
        Prepare the DataLoader for training.

        :param data: Dataset to create DataLoader from.
        :param shuffle: Whether to shuffle the data.
        :return: DataLoader for the dataset.
        """
        return DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def fine_tune(self, layers_to_freeze=None):
        """
        Fine-tune the iDeepLC model on the training dataset.

        :param layers_to_freeze: List of layer names to freeze during fine-tuning.
        :return: Best model based on validation loss.
        """
        LOGGER.info("Starting fine-tuning...")

        if layers_to_freeze:
            self._freeze_layers(layers_to_freeze)

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
        )
        loss_fn = self.loss_function

        if self.validation_data is not None:
            dataloader_train = self.prepare_data(self.train_data, shuffle=True)
            dataloader_val = self.prepare_data(self.validation_data, shuffle=False)
        else:
            train_size = int((1 - self.validation_split) * len(self.train_data))
            val_size = len(self.train_data) - train_size

            if train_size == 0 or val_size == 0:
                raise ValueError(
                    "Training dataset is too small for the requested validation split."
                )

            train_dataset, val_dataset = torch.utils.data.random_split(
                self.train_data, [train_size, val_size]
            )
            dataloader_train = self.prepare_data(train_dataset, shuffle=True)
            dataloader_val = self.prepare_data(val_dataset, shuffle=False)

        LOGGER.info(f"Training on {len(dataloader_train.dataset)} samples.")
        LOGGER.info(f"Validating on {len(dataloader_val.dataset)} samples.")

        best_model = copy.deepcopy(self.model)
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0

            for batch in dataloader_train:
                inputs, target = batch
                inputs = inputs.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                outputs = self.model(inputs.float())
                loss = loss_fn(outputs, target.float().view(-1, 1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            avg_loss = running_loss / len(dataloader_train.dataset)
            LOGGER.info(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_loss:.4f}")

            val_loss, _, _, _ = validate(
                self.model, dataloader_val, loss_fn, self.device
            )

            if val_loss < best_loss:
                best_loss = val_loss
                best_model = copy.deepcopy(self.model)
                patience_counter = 0
                LOGGER.info(f"New best validation loss: {best_loss:.4f}")
            else:
                patience_counter += 1
                LOGGER.info(
                    f"No improvement in validation loss. Patience: {patience_counter}/{self.patience}"
                )

            if patience_counter >= self.patience:
                LOGGER.info("Early stopping triggered.")
                break

        LOGGER.info("Fine-tuning complete.")
        return best_model