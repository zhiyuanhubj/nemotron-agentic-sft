"""Validate FSDP2 parallelism and effective batch size before launching workers."""

import json
from pathlib import Path


def validate_topology(config, world_size, model_path=None):
    sizes = config["distributed"]
    cp, ep = sizes["cp_size"], sizes["ep_size"]
    if world_size < 1 or cp < 1 or ep < 1:
        raise ValueError("World, context-parallel and expert-parallel sizes must be positive")
    if sizes.get("tp_size", 1) != 1 or sizes.get("pp_size", 1) != 1:
        raise ValueError("This recipe supports tp_size=1 and pp_size=1")
    if world_size % cp or world_size % ep:
        raise ValueError("cp_size and ep_size must each divide the total number of workers")
    dp = world_size // cp
    if sizes.get("dp_size") not in (None, dp):
        raise ValueError(f"dp_size must be inferred or equal to {dp}")
    schedule = config["step_scheduler"]
    global_batch, local_batch = schedule["global_batch_size"], schedule["local_batch_size"]
    if local_batch < 1 or global_batch < 1 or global_batch % (local_batch * dp):
        raise ValueError(
            f"global_batch_size must be a positive multiple of local_batch_size * dp_size ({local_batch * dp})"
        )
    if model_path:
        model_config = json.loads((Path(model_path) / "config.json").read_text())
        text_config = model_config.get("text_config", model_config)
        experts = text_config.get("n_routed_experts", text_config.get("num_experts"))
        if experts and experts % ep:
            raise ValueError(f"ep_size must divide the model routed expert count ({experts})")
    return {"world_size": world_size, "dp_size": dp, "cp_size": cp, "ep_size": ep}
