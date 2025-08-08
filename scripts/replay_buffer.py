from pathlib import Path
from tensordict import TensorDict
import torch
from torchrl.data import (
    TensorDictReplayBuffer,
    LazyTensorStorage,
    PrioritizedSampler,
)


device = torch.device("cpu")  # or "cuda", does not matter


def create_replay_buffer():
    return TensorDictReplayBuffer(
        storage=LazyTensorStorage(max_size=10, device=device),
        # storage=LazyMemmapStorage(
        #     max_size=10, scratch_dir=Path("rb_scratch_dir"), device=torch.device("cpu")
        # ),
        sampler=PrioritizedSampler(max_capacity=10, alpha=0.7, beta=0.5),
        batch_size=2,
        priority_key="td_error",
    )


# Create a replay buffer and add some data, WORKS
replay_buffer = create_replay_buffer()
_ = replay_buffer.extend(
    TensorDict(
        {"state": torch.ones(4, 2, dtype=torch.float32, device=device)},
        batch_size=torch.Size((4,)),
    )
)
td = replay_buffer.sample()
td["td_error"] = torch.arange(2, device=device)
replay_buffer.update_tensordict_priority(td)
replay_buffer.dumps(Path("tmp"))
del replay_buffer

# Load the replay buffer directly, WORKS
loaded_replay_buffer = create_replay_buffer()
loaded_replay_buffer.loads(Path("tmp"))

td = loaded_replay_buffer.sample()

# Only load storage, DOES NOT WORK
new_replay_buffer = TensorDictReplayBuffer(
    storage=loaded_replay_buffer.storage,
    sampler=PrioritizedSampler(
        max_capacity=len(loaded_replay_buffer), alpha=0.7, beta=0.5
    ),
    batch_size=2,
    priority_key="td_error",
)

td = new_replay_buffer.sample()

# Also does not work
new_replay_buffer = TensorDictReplayBuffer(
    storage=LazyTensorStorage(max_size=10, device=device),
    sampler=PrioritizedSampler(
        max_capacity=len(loaded_replay_buffer), alpha=0.7, beta=0.5
    ),
    batch_size=2,
    priority_key="td_error",
)
new_replay_buffer.set_storage(loaded_replay_buffer.storage)

td = new_replay_buffer.sample()
