import torch
import hydra
import json
from losses.loss_factory import create_loss
from models.cnn.unet.model import UnetModel
from datasets.wildfire_data_module import WildfireDataModule
from optimizers.optimizer_factory import create_optimizer
from pathlib import Path
from loguru import logger
from omegaconf import DictConfig
from trainers.semantic_segmentation_trainer import SemanticSegmentationTrainer


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig):
    base_folder = Path(cfg["output_path"]) / Path(cfg["run"]["name"])
    base_folder.mkdir(parents=True, exist_ok=True)

    logger.info("Loading split info...")
    with open(Path(cfg["data"]["split_info_file_path"]), "r") as f:
        split_info = json.load(f)

    train_folder_path = Path(split_info["train_folder_path"])
    val_folder_path = Path(split_info["val_folder_path"])
    test_folder_path = Path(split_info["test_folder_path"])
    train_stats = split_info["train_stats"]

    logger.info("Creating data module...")
    data_module = WildfireDataModule(
        input_data_indexes_to_remove=cfg["data"]["input_data_indexes_to_remove"],
        seed=cfg["seed"],
        train_batch_size=cfg["training"]["train_batch_size"],
        eval_batch_size=cfg["training"]["eval_batch_size"],
        source_no_data_value=cfg["data"]["source_no_data_value"],
        destination_no_data_value=cfg["data"]["destination_no_data_value"],
        train_folder_path=train_folder_path,
        val_folder_path=val_folder_path,
        test_folder_path=test_folder_path,
        train_stats=train_stats,
        data_loading_num_workers=cfg["data"]["data_loading_num_workers"],
    )

    data_module.setup()

    train_dl = data_module.train_dataloader()
    val_dl = data_module.val_dataloader()

    model = UnetModel(
        in_channels=cfg["model"]["number_of_input_channels"],
        nb_classes=cfg["model"]["number_of_classes"],
        activation_fn_name=cfg["model"]["activation_fn_name"],
        num_encoder_decoder_blocks=cfg["model"]["num_encoder_decoder_blocks"],
        use_batchnorm=cfg["model"]["use_batchnorm"],
    )

    optimizer = create_optimizer(
        model,
        optimizer_name=cfg["model"]["optimizer"]["name"],
        lr=cfg["model"]["optimizer"]["lr"],
    )

    loss = create_loss(cfg["training"]["loss"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_nb_epochs = cfg["training"]["max_nb_epochs"]
    logger.info(f"Device: {device}")
    logger.info(f"Max number of epochs: {max_nb_epochs}")

    best_model_output_folder = base_folder / "models/"

    trainer = SemanticSegmentationTrainer(
        model=model,
        data_module=data_module,
        optimizer=optimizer,
        device=device,
        loss=loss,
        optimization_metric_name=cfg["training"]["optimization_metric_name"],
        minimize_optimization_metric=cfg["training"]["minimize_optimization_metric"],
        best_model_output_folder=best_model_output_folder,
    )

    train_result = trainer.train_model(
        max_nb_epochs=max_nb_epochs, train_dl=train_dl, val_dl=val_dl
    )

    logger.info(f"Training result: {train_result}")

    train_result_path = base_folder / "train_result.json"
    with open(train_result_path, "w") as f:
        json.dump(train_result, f, indent=4)

    logger.info(f"Training result saved to {str(train_result_path)}")

    logger.success("Training completed successfully!")


if __name__ == "__main__":
    main()
