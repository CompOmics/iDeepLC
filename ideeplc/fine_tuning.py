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

    def __init__(self, model, train_data, loss_function, device="cpu", learning_rate=0.001, epochs=10, batch_size=256,
                  validation_data=None, validation_split=0.1, patience=5):
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
        return DataLoader(data, batch_size=self.batch_size, shuffle=shuffle)

    def fine_tune(self, layers_to_freeze=None):
        """
        Fine-tune the iDeepLC model on the training dataset.

        :param layers_to_freeze: List of layer names to freeze during fine-tuning.
        """
        LOGGER.info("Starting fine-tuning...")
        if layers_to_freeze:
            self._freeze_layers(layers_to_freeze)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn = self.loss_function
        # Prepare DataLoader
        if self.validation_data:
            dataloader_train = self.prepare_data(self.train_data)
            dataloader_val = self.prepare_data(self.validation_data, shuffle=False)
        else:
            # Split the training data into training and validation sets
            train_size = int((1 - self.validation_split) * len(self.train_data))
            val_size = len(self.train_data) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(self.train_data, [train_size, val_size])
            dataloader_train = self.prepare_data(train_dataset)
            dataloader_val = self.prepare_data(val_dataset, shuffle=False)
        LOGGER.info(f"Training on {len(dataloader_train.dataset)} samples.")

        best_model = copy.deepcopy(self.model)
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            for batch in dataloader_train:
                inputs, target = batch
                inputs, target = inputs.to(self.device), target.to(self.device)

                # Forward pass
                outputs = self.model(inputs.float())
                loss = loss_fn(outputs, target.float().view(-1, 1))

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            avg_loss = running_loss / len(dataloader_train.dataset)
            LOGGER.info(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_loss:.4f}")

            # Validate the model after each epoch
            if dataloader_val:
                val_loss, _, _, _ = validate(self.model, dataloader_val, loss_fn, self.device)
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model = copy.deepcopy(self.model)
                    patience_counter = 0
                    LOGGER.info(f"New best validation loss: {best_loss:.4f}")
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    LOGGER.info("Early stopping triggered.")
                    break

        LOGGER.info("Fine-tuning complete.")
        return best_model


