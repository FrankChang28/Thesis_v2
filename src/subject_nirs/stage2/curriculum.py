"""Configure progressive scratch fusion without consuming the shuffled stream."""
def configure_progressive_fusion(model, loaders, config):
    if not config.fusion_ramp_epochs:
        return
    steps = 0
    for loader in loaders:
        if loader.batch_size is not None or loader.num_workers != 0:
            raise ValueError("Progressive fusion expects the original pre-batched loader")
        rows = sum(int(info["n_rows"]) for info in loader.dataset.infos)
        count, remainder = divmod(rows, config.batch_size)
        steps += count + int(bool(remainder) and not loader.dataset.drop_last)
    if not steps:
        raise ValueError("Progressive fusion has no training steps")
    model.fusion_ramp_total_steps = config.fusion_ramp_epochs * steps
    model.fusion_ramp_step = 0
    print(f"Progressive fusion: {config.fusion_ramp_epochs} epoch(s), "
          f"{model.fusion_ramp_total_steps} optimizer steps; validation/test use full fusion")
