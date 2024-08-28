import torch
import tqdm
from tensordict import MemoryMappedTensor, tensorclass
from torch.utils.data import DataLoader
from torchvision import tv_tensors


@tensorclass
class WildfireData:
    images: tv_tensors.Image
    masks: tv_tensors.Mask
    geotransforms: torch.Tensor

    @classmethod
    def from_dataset(cls, dataset, num_workers: int = 2, batch: int = 64):

        if len(dataset) == 0:
            return cls(
                images=tv_tensors.Image(torch.empty((0, 0, 0), dtype=torch.float32)),
                masks=tv_tensors.Mask(torch.empty((0, 0, 0), dtype=torch.int8)),
                geotransforms=torch.empty((0, 0), dtype=torch.float32),
                batch_size=[0],
            )

        data = cls(
            images=tv_tensors.Image(
                MemoryMappedTensor.empty(
                    (
                        len(dataset),
                        *dataset[0][0].shape,
                    ),
                    dtype=torch.float32,
                )
            ),
            masks=tv_tensors.Mask(
                MemoryMappedTensor.empty(
                    (len(dataset), *dataset[0][1].shape), dtype=torch.int8
                )
            ),
            geotransforms=MemoryMappedTensor.empty(
                (len(dataset), *dataset[0][2].shape), dtype=torch.float32
            ),
            batch_size=[len(dataset)],
        )

        data_loader = DataLoader(dataset, batch_size=batch, num_workers=num_workers)
        i = 0
        pbar = tqdm.tqdm(total=len(dataset))
        for images, masks, geotransforms in data_loader:
            _batch = images.shape[0]
            pbar.update(_batch)
            data[i : i + _batch] = cls(
                images=images,
                masks=masks,
                geotransforms=geotransforms,
                batch_size=[_batch],
            )
            i += _batch

        return data
