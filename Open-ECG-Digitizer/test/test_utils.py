import torch

from src.utils import CosineToConstantLR


def test_cosine_to_constant_lr() -> None:
    T_MAX = 10
    LR = 0.1
    ETA_MIN_DIVISOR = 10.0
    eta_min = LR / ETA_MIN_DIVISOR
    eps = 1e-7

    model = torch.nn.Sequential(torch.nn.Linear(10, 10))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = CosineToConstantLR(optimizer, eta_min_divisor=ETA_MIN_DIVISOR, T_max=T_MAX)

    prev_lr = scheduler.get_lr()
    for lr in prev_lr:
        assert abs(lr - LR) < eps

    for _ in range(T_MAX - 1):
        scheduler.step()
        for curr_lr, curr_prev_lr in zip(scheduler.get_lr(), prev_lr):
            assert eta_min < curr_lr < curr_prev_lr

    for _ in range(10):
        scheduler.step()

        for lr in scheduler.get_lr():
            assert abs(lr - eta_min) < eps
