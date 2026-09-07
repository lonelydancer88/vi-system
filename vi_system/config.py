"""价值投资选股系统 —— 配置加载。

设计约束：所有阈值与权重来自 config/rules.yaml，代码内禁止硬编码阈值。
改参数必须新增版本（version + effective_date），禁止原地修改。
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "rules.yaml"


class Config:
    """带版本号与点号路径访问的配置对象。"""

    def __init__(self, data: dict, path: Path | None = None):
        self._data = data
        self.path = path

    # ---- 点号路径访问 -------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        val = self.get(dotted)
        if val is None:
            raise KeyError(f"配置项缺失: {dotted}")
        return val

    def section(self, name: str) -> dict:
        val = self.get(name, {})
        return copy.deepcopy(val) if isinstance(val, dict) else {}

    # ---- 版本信息 -----------------------------------------------------
    @property
    def version(self) -> str:
        return str(self.get("version", "unknown"))

    @property
    def effective_date(self) -> str:
        return str(self.get("effective_date", "unknown"))

    @property
    def stamp(self) -> str:
        return f"v{self.version}@{self.effective_date}"

    # ---- 参数扰动（敏感性检验用） --------------------------------------
    def perturbed(self, dotted: str, factor: float) -> "Config":
        """返回一个新 Config，其中 dotted 指定的数值参数乘以 (1+factor)。"""
        new = Config(copy.deepcopy(self._data), self.path)
        parts = dotted.split(".")
        node = new._data
        for p in parts[:-1]:
            if p not in node:
                return new
            node = node[p]
        last = parts[-1]
        if isinstance(node.get(last), (int, float)):
            node[last] = node[last] * (1.0 + factor)
        return new

    def to_dict(self) -> dict:
        return copy.deepcopy(self._data)


def load_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path else _CONFIG_PATH
    if not p.exists():
        raise FileNotFoundError(f"配置文件不存在: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Config(data, p)
