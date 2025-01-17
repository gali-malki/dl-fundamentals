# Unit 5.5. Organizing Your Data Loaders with Data Modules

import pytorch_lightning as L
from pytorch_lightning.loggers import CSVLogger
import matplotlib.pyplot as plt
import pandas as pd
import torch
from shared_utilities import LightningModel, AmesHousingDataModule, PyTorchMLP
from watermark import watermark

if __name__ == "__main__":

    print(watermark(packages="torch,lightning", python=True))
    print("Torch CUDA available?", torch.cuda.is_available())

    torch.manual_seed(123)
    # Initialize the Ames Housing Data Module
    dm = AmesHousingDataModule()

    # Initialize a PyTorch MLP (Multi-Layer Perceptron) model with 3 features
    pytorch_model = PyTorchMLP(num_features=3)

    # Wrap the PyTorch model in a PyTorch Lightning wrapper for training
    lightning_model = LightningModel(model=pytorch_model, learning_rate=0.001)

    # Configure the PyTorch Lightning trainer
    trainer = L.Trainer(
        max_epochs=30, accelerator="cpu", devices=1, deterministic=True,
        logger=CSVLogger(save_dir="logs/", name="my-model"),
    )
    # start training the model
    trainer.fit(model=lightning_model, datamodule=dm)

    train_mse = trainer.validate(dataloaders=dm.train_dataloader())[0]["val_mse"]
    val_mse = trainer.validate(datamodule=dm)[0]["val_mse"]
    test_mse = trainer.test(datamodule=dm)[0]["test_mse"]
    print(
        f"Train MSE {train_mse:.2f}"
        f" | Val MSE {val_mse:.2f}"
        f" | Test MSE {test_mse:.2f}"
    )

metrics_path = f"{trainer.logger.log_dir}/metrics.csv"
metrics_df = pd.read_csv(metrics_path)

epoch_aggregated = metrics_df.groupby("epoch").mean().reset_index()

epoch_aggregated.plot(x="epoch", y=["train_loss", "val_loss"],
                      grid=True, legend=True, xlabel="Epoch", ylabel="Loss")

epoch_aggregated.plot(x="epoch", y=["train_mse", "val_mse"],
                      grid=True, legend=True, xlabel="Epoch", ylabel="MSE")

plt.show()
