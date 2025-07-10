from pathlib import Path
from tensordict import TensorDict
import torch
from torchrl.data import (
    LazyMemmapStorage,
    TensorDictReplayBuffer,
    LazyTensorStorage,
    PrioritizedSampler,
)


def create_replay_buffer():
    return TensorDictReplayBuffer(
        # storage=LazyTensorStorage(max_size=10, device=torch.device("cpu")),
        storage=LazyMemmapStorage(
            max_size=10, scratch_dir=Path("rb_scratch_dir"), device=torch.device("cpu")
        ),
        # sampler=PrioritizedSampler(max_capacity=10, alpha=0.7, beta=0.5),
        batch_size=2,
        # priority_key="td_error",
    )


replay_buffer = create_replay_buffer()
_ = replay_buffer.extend(
    TensorDict(
        {"state": torch.ones(4, 2, dtype=torch.float32, device=torch.device("cpu"))},
        batch_size=torch.Size((4,)),
    )
)
replay_buffer.dumps(Path("tmp"))
del replay_buffer

new_replay_buffer = create_replay_buffer()
new_replay_buffer.loads(Path("tmp"))
