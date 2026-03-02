from torch.optim.lr_scheduler import LambdaLR


def get_linear_scheduler(optimizer, warmup_steps, training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(training_steps - current_step) / float(max(1, training_steps - warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)