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

    dm = AmesHousingDataModule()

    pytorch_model = PyTorchMLP(num_features=3)

    lightning_model = LightningModel(model=pytorch_model, learning_rate=0.001)

    trainer = L.Trainer(
        max_epochs=30, accelerator="cpu", devices=1, deterministic=True,
        logger=CSVLogger(save_dir="logs/", name="my-model"),
    )
    trainer.fit(model=lightning_model, datamodule=dm)

    train_mse = trainer.validate(dataloaders=dm.train_dataloader())[0]["val_mse"]
    val_mse = trainer.validate(datamodule=dm)[0]["val_mse"]
    test_mse = trainer.test(datamodule=dm)[0]["test_mse"]
    print(
        f"Train MSE {train_mse:.2f}"
        f" | Val MSE {val_mse:.2f}"
        f" | Test MSE {test_mse:.2f}"
    )

# Load metrics from a CSV file
metrics_path = f"{trainer.logger.log_dir}/metrics.csv"
metrics_df = pd.read_csv(metrics_path)

# Aggregate metrics by epoch and calculate the mean for each epoch
epoch_aggregated = metrics_df.groupby("epoch").mean().reset_index()

# Plot training and validation loss
epoch_aggregated.plot(x="epoch", y=["train_loss", "val_loss"],
                      grid=True, legend=True, xlabel="Epoch", ylabel="Loss")

# Plot training and validation mean squared error (MSE)
epoch_aggregated.plot(x="epoch", y=["train_mse", "val_mse"],
                      grid=True, legend=True, xlabel="Epoch", ylabel="MSE")

plt.show()
