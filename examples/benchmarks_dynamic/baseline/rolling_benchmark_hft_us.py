# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pathlib import Path
from typing import Union

import fire
import qlib

from qlib import auto_init
from qlib.contrib.rolling.base import Rolling
from qlib.tests.data import GetData
from qlib.constant import REG_CN, REG_US

DIRNAME = Path(__file__).absolute().resolve().parent


class RollingBenchmark(Rolling):
    # The config in the README.md
    CONF_LIST = [DIRNAME / "workflow_config_linear_Alpha158.yaml", DIRNAME / "workflow_config_lightgbm_Alpha158.yaml"]

    DEFAULT_CONF = CONF_LIST[0]

    def __init__(self, conf_path: Union[str, Path] = DEFAULT_CONF, horizon=1, step = 1, **kwargs) -> None:
        # This code is for being compatible with the previous old code
        conf_path = Path(conf_path)
        super().__init__(conf_path=conf_path, horizon=horizon, step = step, **kwargs)

        for f in self.CONF_LIST:
            if conf_path.samefile(f):
                break
        else:
            self.logger.warning("Model type is not in the benchmark!")


if __name__ == "__main__":
    provider_uri = "~/.qlib/qlib_data/us_data_1min"  # target_dir
    qlib.init(provider_uri=provider_uri, region=REG_US)    
    fire.Fire(RollingBenchmark)
