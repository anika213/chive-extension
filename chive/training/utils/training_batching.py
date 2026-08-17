def compute_per_rank_micro_batch_size(
    *,
    global_batch_size: int,
    world_size: int,
    gradient_accumulation_steps: int,
) -> int:
    assert global_batch_size > 0, "global_batch_size must be positive"
    assert world_size > 0, "world_size must be positive"
    assert gradient_accumulation_steps > 0, "gradient_accumulation_steps must be positive"

    denominator = world_size * gradient_accumulation_steps
    assert global_batch_size % denominator == 0, (
        f"global_batch_size {global_batch_size} must be divisible by "
        "world_size * gradient_accumulation_steps "
        f"({world_size} * {gradient_accumulation_steps} = {denominator})"
    )
    return global_batch_size // denominator
