import tempfile

from tensordict import TensorDict
import torch
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer

size = 100

data = TensorDict(
    {
        "a": torch.arange(12).view(3, 4),
        ("b", "c"): torch.arange(15).view(3, 5),
    },
    batch_size=[3],
)
with tempfile.TemporaryDirectory() as tempdir:
    buffer_lazymemmap = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(size, scratch_dir=tempdir), batch_size=12
    )
    buffer_lazymemmap.extend(data)
    print(f"The buffer has {len(buffer_lazymemmap)} elements")
    sample = buffer_lazymemmap.sample()
    print("sample:", sample)
    del buffer_lazymemmap
