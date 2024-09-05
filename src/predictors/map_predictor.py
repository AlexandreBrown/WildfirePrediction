import torch
import torch.nn as nn
import torch.nn.functional as F
import asyncio
from osgeo import gdal
from osgeo import osr
from torch.utils.data.dataloader import DataLoader
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from raster_io.read import get_extension
from boundaries.canada_boundary import CanadaBoundary


class MapPredictor:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        output_folder_path: Path,
        canada_boundary: CanadaBoundary,
        convert_model_output_to_probabilities: bool = True,
        data_srid: int = 3978,
    ):
        self.model = model.to(device)
        self.device = device
        self.output_folder_path = output_folder_path
        self.canada_boundary = canada_boundary
        self.use_probabilities = convert_model_output_to_probabilities
        self.output_folder_path.mkdir(parents=True, exist_ok=True)
        self.data_srid = data_srid

    async def predict(self, dataloader: DataLoader) -> Path:
        logger.info("Predicting...")

        self.predict_dl = dataloader
        self.predictions_saved = 0

        self.model.eval()

        predict_loader = tqdm(self.predict_dl, desc="Predictions", leave=False)

        predictions_files = []

        with torch.no_grad():
            for predict_data in predict_loader:

                images = predict_data.images.to(self.device)
                geotransforms = predict_data.geotransforms

                y_hat = self.model(images)
                y_hat = torch.squeeze(y_hat, dim=1)

                if self.use_probabilities:
                    y_hat = F.logsigmoid(y_hat).exp()

                for i, prediction in enumerate(y_hat):
                    geotransform = geotransforms[i]
                    predictions_files.append(
                        self.save_prediction(prediction, geotransform)
                    )
                    self.predictions_saved += 1

        logger.success(f"Generated {self.predictions_saved} predictions!")

        return await self.merge_predictions_files(predictions_files)

    def save_prediction(
        self, prediction: torch.Tensor, geotransform: torch.Tensor
    ) -> Path:
        driver = gdal.GetDriverByName("GTiff")
        output_file = Path(self.output_folder_path) / Path(
            f"{self.predictions_saved}{get_extension('GTiff')}"
        )
        height, width = prediction.shape[0], prediction.shape[1]
        output_dataset = driver.Create(
            str(output_file),
            width,
            height,
            1,
            gdal.GDT_Float32,
        )
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(self.data_srid)
        output_dataset.SetProjection(srs.ExportToWkt())
        output_dataset.SetGeoTransform(geotransform.cpu().numpy())
        output_band = output_dataset.GetRasterBand(1)
        output_band.WriteArray(prediction.cpu().numpy())
        output_dataset.FlushCache()

        return output_file

    async def merge_predictions_files(self, predictions_files: list) -> Path:
        logger.info("Merging predictions files...")

        input_files = [str(f) for f in predictions_files]
        vrt_file = Path(self.output_folder_path) / Path("merged") / "merged.vrt"
        vrt_file.parent.mkdir(parents=True, exist_ok=True)

        await self.run_command(
            "gdalbuildvrt",
            [
                "-overwrite",
                str(vrt_file),
                *input_files,
            ],
        )

        merged_predictions_output_file = vrt_file.with_suffix(get_extension("gtiff"))
        await self.run_command(
            "gdal_translate",
            [
                "-of",
                "GTiff",
                str(vrt_file),
                str(merged_predictions_output_file),
            ],
        )

        for file in predictions_files:
            file.unlink()

        vrt_file.unlink()

        canada_boundary_file = self.canada_boundary.save(self.output_folder_path)

        logger.info("Clipping merged predictions to Canada boundary...")
        clipped_merged_predictions_output_file = (
            merged_predictions_output_file.with_stem("predictions_output_clipped")
        )
        params = [
            "--quiet",
            "-overwrite",
            "-multi",
            "-cutline_srs",
            f"EPSG:{self.canada_boundary.target_epsg}",
            "-cutline",
            f"{str(canada_boundary_file)}",
            "-crop_to_cutline",
            "-dstnodata",
            "0",
            "-of",
            "GTiff",
            str(merged_predictions_output_file),
            str(clipped_merged_predictions_output_file),
        ]

        await self.run_command(
            "gdalwarp",
            params,
        )

        merged_predictions_output_file.unlink()
        canada_boundary_file.unlink()

        return clipped_merged_predictions_output_file

    async def run_command(self, program: str, commands: list):
        proc = await asyncio.create_subprocess_exec(
            program,
            *commands,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()

        error = stderr.decode()
        if error != "":
            logger.error(error)
            raise Exception(error)
