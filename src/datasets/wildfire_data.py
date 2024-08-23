import torch
import tqdm
from tensordict import MemoryMappedTensor, tensorclass
from torch.utils.data import DataLoader


@tensorclass
class WildfireData:
    images: torch.Tensor
    masks: torch.Tensor
    geotransforms: torch.Tensor

    @classmethod
    def from_dataset(cls, dataset, num_workers: int = 2, batch: int = 64):

        if len(dataset) == 0:
            return cls(
                images=torch.empty((0, 0, 0), dtype=torch.float32),
                masks=torch.empty((0, 0, 0), dtype=torch.long),
                geotransforms=torch.empty((0, 0), dtype=torch.float32),
                batch_size=[0],
            )

        data = cls(
            images=MemoryMappedTensor.empty(
                (
                    len(dataset),
                    *dataset[0][0].shape,
                ),
                dtype=torch.float32,
            ),
            masks=MemoryMappedTensor.empty(
                (len(dataset), *dataset[0][1].shape), dtype=torch.long
            ),
            geotransforms=MemoryMappedTensor.empty(
                (len(dataset), *dataset[0][2].shape), dtype=torch.float32
            ),
            batch_size=[len(dataset)],
        )
        # locks the tensorclass and ensures that is_memmap will return True.
        data.memmap_()

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
