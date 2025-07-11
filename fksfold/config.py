
# Shared mutable configuration dictionary. 任何模块 `import fksfold.config as cfg` 后
# 都将拿到同一份 `global_config` 对象，可用来在运行过程中跨模块读写配置参数。
# 只放一些运行时需要动态修改或全局可调的简单标量，不要把复杂对象塞进来。

from __future__ import annotations

from typing import Any, Dict, Optional


# 默认值。可在运行时通过 `update_global_config` 修改或自行直接赋值。
global_config: Dict[str, Any] = {
    # 当前扩散步长 sigma，用于在不同模块之间共享进度信息；初始化为 None。
    "current_sigma": None,
    # 触发 RMSD 约束的 sigma 阈值；当 `current_sigma < rmsd_sigma_threshold` 时生效。
    "rmsd_sigma_threshold": 1.0,
    # 触发 FK (frame/ptm) 约束的 sigma 阈值；当 `current_sigma < fk_sigma_threshold` 时生效。
    "fk_sigma_threshold": 1.0,
}


def update_global_config(**kwargs: Any) -> None:
    """Update entries inside :data:`global_config` in-place.

    Example::

        from fksfold.config import update_global_config
        update_global_config(rmsd_sigma_threshold=0.8, fk_sigma_threshold=0.5)
    """

    for k, v in kwargs.items():
        global_config[k] = v