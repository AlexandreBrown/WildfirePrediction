import torch
import hydra
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
    model = UnetModel(
        in_channels=cfg["model"]["number_of_input_channels"],
        nb_classes=cfg["model"]["number_of_classes"],
        activation_fn_name=cfg["model"]["activation_fn_name"],
        num_encoder_decoder_blocks=cfg["model"]["num_encoder_decoder_blocks"],
        use_batchnorm=cfg["model"]["use_batchnorm"],
    )

    base_folder = Path("trainings") / Path(cfg["run"]["name"])
    base_folder.mkdir(parents=True, exist_ok=True)

    data_output_folder = base_folder / "data/"

    train_periods = get_train_periods(cfg)

    input_data_periods_folders_paths = [
        Path(p) for p in cfg["data"]["input_data_periods_folders_paths"]
    ]
    target_periods_folders_paths = [
        Path(p) for p in cfg["data"]["target_periods_folders_paths"]
    ]

    data_module = WildfireDataModule(
        input_data_periods_folders_paths=input_data_periods_folders_paths,
        input_data_indexes_to_remove=list(cfg["data"]["input_data_indexes_to_remove"]),
        target_periods_folders_paths=target_periods_folders_paths,
        train_periods=train_periods,
        output_folder_path=data_output_folder,
        seed=cfg.seed,
        train_batch_size=cfg["training"]["train_batch_size"],
        eval_batch_size=cfg["training"]["eval_batch_size"],
        val_split=cfg["training"]["val_split"],
        model_input_size=cfg["model"]["input_resolution_in_pixels"],
        num_workers=cfg["training"]["num_workers"],
        source_no_data_value=cfg["data"]["source_no_data_value"],
        destination_no_data_value=cfg["data"]["destination_no_data_value"],
    )

    data_module.setup(stage="fit")
    train_dl = data_module.train_dataloader()
    val_dl = data_module.val_dataloader()

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
    logger.success("Training completed successfully!")


def get_train_periods(cfg: DictConfig) -> list:
    target_year_start_inclusive = cfg["training"]["train_periods"]["start_inclusive"]
    target_year_end_inclusive = cfg["training"]["train_periods"]["end_inclusive"]
    target_period_length_in_years = cfg["training"]["train_periods"][
        "period_length_in_years"
    ]
    target_years_ranges = []

    for target_year_start in range(
        target_year_start_inclusive, target_year_end_inclusive + 1, 1
    ):
        target_year_end = target_year_start + target_period_length_in_years - 1
        assert (
            target_year_end <= target_year_end_inclusive
        ), f"Target year end {target_year_end} is greater than target year end inclusive {target_year_end_inclusive}"
        target_years_ranges.append(range(target_year_start, target_year_end + 1))

    logger.info(f"Target years ranges: {target_years_ranges}")

    return target_years_ranges


if __name__ == "__main__":
    main()
