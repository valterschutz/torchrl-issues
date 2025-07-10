import tempfile

from tensordict import TensorDict
import torch
from torchrl.data import (
    LazyMemmapStorage,
    LazyTensorStorage,
    PrioritizedSampler,
    ReplayBuffer,
    TensorDictReplayBuffer,
)

size = 100

data = TensorDict(
    {
        "a": torch.arange(12).view(3, 4),
        ("b", "c"): torch.arange(15).view(3, 5),
    },
    batch_size=[3],
)
with tempfile.TemporaryDirectory() as tempdir:
    replay_buffer = TensorDictReplayBuffer(
        # storage=LazyMemmapStorage(size, scratch_dir=tempdir),
        sampler=PrioritizedSampler(
            max_capacity=size,
            alpha=0.6,
            beta=0.4,
        ),
        storage=LazyTensorStorage(size),
        batch_size=12,
    )
    replay_buffer.extend(data)
    print(f"The buffer has {len(replay_buffer)} elements")
    sample = replay_buffer.sample()
    print("sample:", sample)
    replay_buffer.dumps(tempdir)
