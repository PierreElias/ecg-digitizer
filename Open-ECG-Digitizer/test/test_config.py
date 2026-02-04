from src.config.default import get_cfg
from src.train import main


def test_unet() -> None:
    cfg = get_cfg("./test/test_data/config/unet.yml")
    main(cfg)
