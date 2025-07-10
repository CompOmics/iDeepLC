def get_config(lr=1e-3, epoch=10, batch=256, kernel=5, kernel2=3, kernel3=9, kernel4=7, cnn_channels=245,
               cnn2_channels=41, cnn3_channels=35, cnn4_channels=50, cnn_layers=1, cnn2_layers=0, cnn3_layers=5,
               cnn4_layers=3, fc_layers=2, fc_output=78, fc2_layers=1, fc2_output=77, drop=0.23, clip=0.25, layers_to_freeze=None):
    """
    Initialize the configuration for the model hyperparameters

    :param lr: learning rate
    :param epoch: epochs
    :param batch: batch size
    :param kernel: kernel size for the first CNN layer
    :param kernel2: kernel size for the second CNN layer
    :param kernel3: kernel size for the third CNN layer
    :param kernel4: kernel size for the fourth CNN layer
    :param cnn_channels: number of channels for the first CNN layer
    :param cnn2_channels: number of channels for the second CNN layer
    :param cnn3_channels: number of channels for the third CNN layer
    :param cnn4_channels: number of channels for the fourth CNN layer
    :param cnn_layers: number of layers for the first CNN layer
    :param cnn2_layers: number of layers for the second CNN layer
    :param cnn3_layers: number of layers for the third CNN layer
    :param cnn4_layers: number of layers for the fourth CNN layer
    :param fc_layers: number of layers for the first fully connected layer
    :param fc_output: output size for the first fully connected layer
    :param fc2_layers: number of layers for the second fully connected layer
    :param fc2_output: output size for the second fully connected layer
    :param drop: dropout rate
    :param clip: clipping size
    :param layers_to_freeze: list of layer names to freeze during finetuning
    :return: configuration dictionary
    """

    config = {"learning_rate": lr, "epochs": epoch, "batch_size": batch, "kernel_size": kernel, "kernel2_size": kernel2,
              "kernel3_size": kernel3, "kernel4_size": kernel4, "fc_out": fc_output, "fc_layers": fc_layers,
              "fc2_out": fc2_output, "fc2_layers": fc2_layers, "cnn_layers": cnn_layers, "cnn_channels": cnn_channels,
              "cnn2_layers": cnn2_layers, "cnn2_channels": cnn2_channels, "cnn3_layers": cnn3_layers,
              "cnn3_channels": cnn3_channels, "cnn4_layers": cnn4_layers, "cnn4_channels": cnn4_channels,
              "clipping_size": clip, "dropout": drop, "layers_to_freeze": layers_to_freeze}

    return config
