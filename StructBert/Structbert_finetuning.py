import os.path as osp

import torch
from modelscope.trainers import build_trainer
from modelscope.msdatasets import MsDataset
from modelscope.utils.hub import read_config
from modelscope.metainfo import Metrics
from modelscope.hub.api import HubApi

api = HubApi()
api.login('your_modelscope_loginApi')
# 都改为自己的模型位置和数据集名称
model_id = 'iic/nlp_structbert_sentiment-classification_chinese-base'
dataset_id = 'yjsmh'
# 训练日志、模型保存的工作目录
WORK_DIR = 'workspace'
# 使用数据集训练几次，可以自行调整，尽量在2，3中，过多会导致过拟合
max_epochs = 8


def cfg_modify_fn(cfg):
    cfg.train.max_epochs = max_epochs
    cfg.train.hooks = [
        # 100轮打印一次日志，数据小可以降低
        {

            'type': 'TextLoggerHook',
            'interval': 100
        },
        # 多少轮保存一次checkpoint
        {
            "type": "CheckpointHook",
            "interval": 1
        }]
    # 评估指标，默认返回的是准确率
    cfg.evaluation.metrics = [Metrics.seq_cls_metric]
    cfg['dataset'] = {
        'train': {
            # 数据集中标签的名字
            'labels': ['0', '1'],
            # 训练文档中训练的数据（句子）的列名
            'first_sequence': 'sentence',
            # 训练文档中标签那一列的名字
            'label': 'label',
        }
    }
    # 初始学习率
    cfg.train.lr_scheduler = {'type': 'CosineAnnealingLR','T_max':4,
                              'options': {'warmup': {'type': 'LinearWarmup', 'warmup_iters': 4}}}
    return cfg


if __name__ == '__main__':
    # dataset_id为数据集名，namesapce改为为个人账号名；split指定加载的是训练or测试or验证集
    train_dataset = MsDataset.load(dataset_id,namespace='SCUzqy',  split='train')
    eval_dataset = MsDataset.load(dataset_id,namespace='SCUzqy', split='validation')

    # 去除训练集和验证集中标签和句子为空的部分
    train_dataset = train_dataset.filter(lambda x: x["label"] is not None and x["sentence"] is not None)
    eval_dataset = eval_dataset.filter(lambda x: x["label"] is not None and x["sentence"] is not None)

    # 将标签转化成012的对应关系
    def map_labels(examples):
        map_dict = {0: "1", 1: "1"}
        examples['label'] = map_dict[int(examples['label'])]
        return examples


    # 调用函数将训练集和验证集的标签转化
    train_dataset = train_dataset.map(map_labels)
    eval_dataset = eval_dataset.map(map_labels)
    # 训练参数model模型名，train_dataset训练数据集名称，eval_dataset验证集数据集名称，work_dir为训练日志、模型保存的工作目录，cfg_modify_fn调用函数修改配置文件按上述修改
    kwargs = dict(
        model=model_id,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        work_dir=WORK_DIR,
        cfg_modify_fn=cfg_modify_fn)
    # 构建trainer，使用nlp-base-trainer
    trainer = build_trainer(name='nlp-base-trainer', default_args=kwargs)
    print('===============================================================')
    print('预训练模型加载，训练开始')
    print('===============================================================')
    # 开始训练，结束打印训练成功
    trainer.train()
    print('===============================================================')
    print('训练成功')
    print('===============================================================')
    # 验证验证集
    # for i in range(max_epochs):
    eval_results = trainer.evaluate()
    print(f'epoch {1} evaluation result:')

    print('===============================================================')
    print('验证结束')
    print('===============================================================')
