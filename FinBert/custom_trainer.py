"""
===========================================================
custom_trainer.py

自定义 HuggingFace Trainer

作用：

1. 与 ModelScope Trainer 保持一致
2. 自动输出每个Epoch训练Loss
3. 自动输出验证Loss
4. 自动输出当前Learning Rate
5. 保留 HuggingFace 全部训练功能
6. 后续方便扩展（AUC、FocalLoss、Class Weight等）

Author:
===========================================================
"""

import numpy as np
import torch

from transformers import Trainer
from transformers.trainer_utils import EvalPrediction
from transformers import EarlyStoppingCallback


class CustomTrainer(Trainer):
    """
    自定义 Trainer

    在 HuggingFace Trainer 基础上增加：

    ① 更漂亮的日志输出
    ② 当前学习率输出
    ③ 后续方便增加 Loss
    """

    def log(self, logs, start_time=None):
        """
        每次 logging 时调用

        相当于 ModelScope 的 TextLoggerHook
        """

        if "loss" in logs:

            print(
                f"[Train]"
                f" Step={self.state.global_step:<6}"
                f" Loss={logs['loss']:.6f}"
            )

        if "eval_loss" in logs:

            print(
                f"[Validation]"
                f" Epoch={self.state.epoch:.2f}"
                f" EvalLoss={logs['eval_loss']:.6f}"
            )

        if "learning_rate" in logs:

            print(
                f" LearningRate={logs['learning_rate']:.8f}"
            )

        super().log(logs, start_time)

    ####################################################################
    # 后续如果需要修改Loss，可以重写compute_loss()
    ####################################################################

    def compute_loss(
            self,
            model,
            inputs,
            return_outputs=False,
            **kwargs
    ):
        """
        默认交叉熵Loss

        与 BertForSequenceClassification 完全一致。

        后续如果需要：

            class weight

            focal loss

            label smoothing

        只需要修改这里即可。
        """

        outputs = model(**inputs)

        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

    ####################################################################
    # prediction_step保持官方实现
    ####################################################################

    def prediction_step(
            self,
            model,
            inputs,
            prediction_loss_only=False,
            ignore_keys=None
    ):
        """
        保持官方实现。

        以后如果需要：

            输出Attention

            输出Hidden State

            输出Embedding

        可以在这里修改。
        """

        return super().prediction_step(
            model,
            inputs,
            prediction_loss_only,
            ignore_keys
        )
