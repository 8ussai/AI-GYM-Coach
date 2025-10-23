# modules/runtime/pipelines/base_pipeline.py
from __future__ import annotations
from typing import Union
from modules.common.workers import BaseModelWorker


class BasePipeline:
    """
    واجهة عامة لأي Pipeline تشغيل.
    """
    def __init__(self, source: Union[int, str], settings):
        self.source = source
        self.settings = settings

    def build_worker(self) -> BaseModelWorker:
        """
        يبني ويرجع Worker يورّث من BaseModelWorker.
        """
        raise NotImplementedError
