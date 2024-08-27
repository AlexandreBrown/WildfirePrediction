from comet_ml.exceptions import InterruptedExperiment
import asyncio
import torch
import hydra
import json
from models.cnn.unet.model import UnetModel
from datasets.wildfire_data_module import WildfireDataModule
from pathlib import Path
from loguru import logger
from omegaconf import DictConfig
from predictors.map_predictor import MapPredictor
from logging_utils.logging import setup_logger


@hydra.main(version_base=None, config_path="config", config_name="predict")
def main(cfg: DictConfig):
    run_name = cfg["run"]["name"]
    debug = cfg["debug"]
    setup_logger(logger, run_name, debug)
    logger.info(f"Run name: {run_name}")
    logger.info(f"Debug : {debug}")

    run_output_path = Path(cfg["output_path"]) / Path(cfg["run"]["name"])
    run_output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading split info...")
    with open(Path(cfg["data"]["split_info_file_path"]), "r") as f:
        split_info = json.load(f)

    train_stats = split_info["train_stats"]
    predict_input_data_folder_path = Path(cfg["data"]["input_data_folder_path"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    logger.info("Creating data module...")
    data_module = WildfireDataModule(
        input_data_indexes_to_remove=cfg["data"]["input_data_indexes_to_remove"],
        eval_batch_size=cfg["predict"]["batch_size"],
        input_data_no_data_value=cfg["data"]["input_data_no_data_value"],
        input_data_new_no_data_value=cfg["data"]["input_data_new_no_data_value"],
        predict_folder_path=predict_input_data_folder_path,
        train_stats=train_stats,
        data_loading_num_workers=cfg["data"]["data_loading_num_workers"],
        device=device,
    )

    logger.info("Setting up data module...")
    data_module.setup(stage="predict")

    predict_dl = data_module.predict_dataloader()

    model = UnetModel(
        in_channels=cfg["model"]["number_of_input_channels"],
        nb_classes=cfg["model"]["number_of_classes"],
        activation_fn_name=cfg["model"]["activation_fn_name"],
        num_encoder_decoder_blocks=cfg["model"]["num_encoder_decoder_blocks"],
        use_batchnorm=cfg["model"]["use_batchnorm"],
    )

    logger.info("Loading trained model...")
    model.load_state_dict(
        torch.load(Path(cfg["model"]["trained_model_path"]), weights_only=True),
        strict=True,
    )

    convert_model_output_to_probabilities = cfg["predict"][
        "convert_model_output_to_probabilities"
    ]
    predictor = MapPredictor(
        model=model,
        device=device,
        output_folder_path=run_output_path,
        convert_model_output_to_probabilities=convert_model_output_to_probabilities,
    )

    try:
        asyncio.run(predictor.predict(predict_dl))
    except InterruptedExperiment as exc:
        logger.info("status", str(exc))
        logger.info("Experiment interrupted!")


if __name__ == "__main__":
    main()
