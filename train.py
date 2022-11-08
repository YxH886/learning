diff --git a/research/cv/CBAM/README.md b/research/cv/CBAM/README.md
new file mode 100644
index 0000000000000000000000000000000000000000..f96a699a2f468456325b9e59865e026b4d7c7acd
--- /dev/null
+++ b/research/cv/CBAM/README.md
@@ -0,0 +1,164 @@
+# 目录
+
+- [cbam说明](#cbam说明)
+- [模型架构](#模型架构)
+- [数据集](#数据集)
+- [环境要求](#环境要求)
+- [脚本说明](#脚本说明)
+    - [脚本及样例代码](#脚本及样例代码)
+    - [脚本参数](#脚本参数)
+    - [训练过程](#训练过程)
+        - [用法](#训练用法)
+    - [评估过程](#评估过程)
+        - [用法](#评估用法)
+        - [结果](#评估结果)
+- [模型描述](#模型描述)
+    - [性能](#性能)
+        - [训练性能](#训练性能)
+- [随机情况说明](#随机情况说明)
+- [ModelZoo主页](#modelzoo主页)
+
+# CBAM说明
+
+CBAM(Convolutional Block Attention Module)是一种轻量级注意力模块的提出于2018年，它可以在空间维度和通道维度上进行Attention操作。
+
+[论文](https://arxiv.org/abs/1807.06521)：  Sanghyuan Woo, Jongchan Park, Joon-Young Lee, In So Kweon. CBAM: Convolutional Block Attention Module.
+
+# 模型架构
+
+CBAM整体网络架构如下：
+
+[链接](https://arxiv.org/abs/1807.06521)
+
+# 数据集
+
+使用的数据集：[RML2016.10A](https://www.xueshufan.com/publication/2562146178)
+
+- 数据集大小：共611M，总共有22万条样本。
+    - 训练集：110000条样本。
+    - 测试集：取一个信噪比下的数据，含有样本5500条。
+- 数据格式：IQ（In-phaseand Quadrature）：2*128。
+    - 注：数据在src/data中处理。
+
+# 环境要求
+
+- 硬件（Ascend）
+    - 使用Ascend处理器来搭建硬件环境。
+- 框架
+    - [MindSpore](https://www.mindspore.cn/install)
+- 如需查看详情，请参见如下资源：
+    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
+    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)
+
+# 脚本说明
+
+## 脚本及样例代码
+
+```path
+.
+└─CBAM
+  ├─src
+    ├─data.py                       # 数据集处理
+    ├─model.py                      # CBAM网络定义
+    ├─get_lr.py                     # 生成学习率
+    ├─model_utils
+      ├─config.py                   # 参数配置
+      ├─device_adapter.py           # 适配云上或线下
+      ├─local_adapter.py            # 线下配置
+      ├─moxing_adapter.py           # 云上配置
+  ├──eval.py                        # 评估网络
+  ├──train.py                       # 训练网络
+  ├──default_config.yaml            # 参数配置
+  └──README.md                      # README文件
+```
+
+## 脚本参数
+
+在default_config.yaml中可以同时配置训练和评估参数。
+
+```python
+"batch_size":32,                   # 输入张量的批次大小
+"epoch_size":70,                   # 训练周期大小
+"lr_init":0.001,                   # 初始学习率
+"save_checkpoint":True,            # 是否保存检查点
+"save_checkpoint_epochs":1,        # 两个检查点之间的周期间隔；默认情况下，最后一个检查点将在最后一个周期完成后保存
+"keep_checkpoint_max":10,          # 只保存最后一个keep_checkpoint_max检查点
+"warmup_epochs":5,                 # 热身周期
+```
+
+## 训练过程
+
+### 训练用法
+
+首先需要在`default_config.yaml`中设置好超参数。
+
+您可以通过华为云等资源开始训练，其中配置如下所示：
+
+```shell
+Ascend:
+   训练输入：data_url = /cbam/dataset,
+   训练输出：train_url = /cbam/train_output,
+   输出日志：/cbam/train_logs
+```
+
+### 训练结果
+
+Ascend评估结果保存在`/cbam/train_logs`下。您可以在日志中找到类似以下的结果。
+
+```log
+epoch: 1 step: 3437, loss is 0.7258548
+epoch: 2 step: 3437, loss is 0.6980165
+epoch: 3 step: 3437, loss is 0.6887816
+epoch: 4 step: 3437, loss is 0.7017617
+epoch: 5 step: 3437, loss is 0.694684
+```
+
+## 评估过程
+
+### 评估用法
+
+与训练相同，在`default_config.yaml`中设置好超参数，通过华为云平台进行训练：
+
+```shell
+Ascend:
+训练输入：data_url = /cbam/dataset,
+训练输入：ckpt_file = /cbam/train_output/cbam_train-70_3437.ckpt,
+输出日志：/cbam/eval_logs
+```
+
+### 评估结果
+
+Ascend评估结果保存在`/cbam/eval_logs`下。您可以在日志中找到类似以下的结果。
+
+```log
+result: {'Accuracy': 0.8494152046783626}
+```
+
+# 模型描述
+
+## 性能
+
+### 训练性能
+
+| 参数                       | Ascend 910                                                  |
+| -------------------------- | ---------------------------------------------------------- |
+| 资源                       | Ascend 910                                                  |
+| 上传日期                   | 2022-5-31                                                    |
+| MindSpore版本              | 1.5.1                                                       |
+| 数据集                     | RML2016.10A                                                 |
+| 训练参数                   | default_config.yaml                                          |
+| 优化器                     | Adam                                                         |
+| 损失函数                   | BCEWithLogitsLoss                                             |
+| 损失                       |  0.6702158                                                    |
+| 准确率                     | 84.9%                                                         |
+| 总时长                     | 41分钟 （1卡）                                              |
+| 调优检查点                 | 5.80 M（.ckpt文件）                                              |
+
+# 随机情况说明
+
+在train.py中的随机种子。
+
+# ModelZoo主页
+
+请浏览官网[主页](https://gitee.com/mindspore/models)。
+
diff --git a/research/cv/CBAM/default_config.yaml b/research/cv/CBAM/default_config.yaml
new file mode 100644
index 0000000000000000000000000000000000000000..073d6f2eeed598d680ac4dd3a4c78244b3a88eba
--- /dev/null
+++ b/research/cv/CBAM/default_config.yaml
@@ -0,0 +1,67 @@
+# Builtin Configurations(DO NOT CHANGE THESE CONFIGURATIONS unless you know exactly what you are doing)
+enable_modelarts: True
+data_url: ""
+train_url: ""
+checkpoint_url: ""
+data_path: "/cache/data/"
+output_path: "/cache/train"
+load_path: "/cache/checkpoint_path"
+checkpoint_path: './checkpoint/'
+checkpoint_file: './checkpoint/cbam_train-70_3437.ckpt'
+device_target: Ascend
+enable_profiling: False
+
+ckpt_path: "/cache/train"
+ckpt_file: "/cache/train//cbam_train-70_3437.ckpt"
+# ==============================================================================
+# Training options
+lr: 0.01
+lr_init: 0.01
+lr_max: 0.1
+lr_epochs: '30, 60, 90, 120'
+lr_scheduler: "piecewise"
+warmup_epochs: 5
+epoch_size: 70
+max_epoch: 70
+momentum: 0.9
+loss_scale: 1.0
+label_smooth: 0
+label_smooth_factor: 0
+weight_decay: 0.0005
+batch_size: 32
+keep_checkpoint_max: 10
+MINDIR_name: 'cbam.MINDIR'
+ckpt_file_path: './cbam_train-70_3437.ckpt'
+num_classes : 4
+dataset_name: 'rate'
+dataset_sink_mode: True
+device_id: 0
+save_checkpoint: True
+save_checkpoint_epochs: 1
+local_data_path: '../data'
+run_distribute: False
+batch_norm: False
+initialize_mode: "KaimingNormal"
+padding: 1
+pad_mode: 'pad'
+has_bias: False
+has_dropout: True
+
+# Model Description
+model_name: CBAM
+file_name: 'CBAM'
+file_format: 'MINDIR'
+
+---
+# Config description for each option
+enable_modelarts: 'Whether training on modelarts, default: False'
+data_url: 'Dataset url for obs'
+train_url: 'Training output url for obs'
+data_path: 'Dataset path for local'
+output_path: 'Training output path for local'
+
+device_target: 'Target device type'
+enable_profiling: 'Whether enable profiling while training, default: False'
+
+---
+device_target: ['Ascend', 'GPU', 'CPU']
diff --git a/research/cv/CBAM/eval.py b/research/cv/CBAM/eval.py
new file mode 100644
index 0000000000000000000000000000000000000000..ff9713ea208137f64bfcd9d9c2c1ef40dd40a3b9
--- /dev/null
+++ b/research/cv/CBAM/eval.py
@@ -0,0 +1,70 @@
+# Copyright 2022 Huawei Technologies Co., Ltd
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+# http://www.apache.org/licenses/LICENSE-2.0
+#
+# Unless required by applicable law or agreed to in writing, software
+# distributed under the License is distributed on an "AS IS" BASIS,
+# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
+# See the License for the specific language governing permissions and
+# limitations under the License.
+# ============================================================================
+
+"""
+##############test CBAM example on dataset#################
+python eval.py
+"""
+
+from mindspore import context
+from mindspore.train import Model
+from mindspore.communication.management import init
+from mindspore.train.serialization import load_checkpoint, load_param_into_net
+from mindspore.nn.metrics import Accuracy
+from mindspore.nn import SoftmaxCrossEntropyWithLogits
+
+from src.model_utils.config import config
+from src.model_utils.moxing_adapter import moxing_wrapper
+from src.model_utils.device_adapter import get_device_id, get_device_num
+from src.data import create_dataset
+from src.model import resnet50_cbam
+
+
+def modelarts_process():
+    config.ckpt_path = config.ckpt_file
+
+
+@moxing_wrapper(pre_process=modelarts_process)
+def eval_():
+    """ model eval """
+    device_num = get_device_num()
+    if device_num > 1:
+        context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', save_graphs=False)
+        if config.device_target == "Ascend":
+            context.set_context(device_id=get_device_id())
+            init()
+        elif config.device_target == "GPU":
+            init()
+    print("================init finished=====================")
+    ds_eval = create_dataset(data_path=config.data_path, batch_size=config.batch_size,
+                             training=False, snr=2, target=config.device_target)
+    if ds_eval.get_dataset_size() == 0:
+        raise ValueError("Please check dataset size > 0 and batch_size <= dataset size.")
+    print("ds_eval_size", ds_eval.get_dataset_size())
+    print("===============create dataset finished==============")
+
+    net = resnet50_cbam(phase="test")
+    param_dict = load_checkpoint(config.ckpt_path)
+    print("load checkpoint from [{}].".format(config.ckpt_path))
+    load_param_into_net(net, param_dict)
+    net.set_train(False)
+    loss = SoftmaxCrossEntropyWithLogits()
+    model = Model(net, loss_fn=loss, metrics={"Accuracy": Accuracy()})
+    result = model.eval(ds_eval, dataset_sink_mode=False)
+    print("===================result: {}==========================".format(result))
+
+
+if __name__ == '__main__':
+    eval_()
diff --git a/research/cv/CBAM/src/data.py b/research/cv/CBAM/src/data.py
new file mode 100644
index 0000000000000000000000000000000000000000..960b517259f2833f5ff45ebc33f622ea539d8531
--- /dev/null
+++ b/research/cv/CBAM/src/data.py
@@ -0,0 +1,151 @@
+# Copyright 2022 Huawei Technologies Co., Ltd
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+# http://www.apache.org/licenses/LICENSE-2.0
+#
+# Unless required by applicable law or agreed to in writing, software
+# distributed under the License is distributed on an "AS IS" BASIS,
+# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
+# See the License for the specific language governing permissions and
+# limitations under the License.
+# ============================================================================
+
+"""Data preprocessing"""
+
+import os
+import pickle
+import numpy as np
+
+from mindspore.communication.management import get_rank, get_group_size
+import mindspore.dataset as de
+import mindspore.common.dtype as mstype
+import mindspore.dataset.transforms.c_transforms as C
+
+
+def _get_rank_info(run_distribute):
+    """get rank size and rank id"""
+    rank_size = int(os.environ.get("RANK_SIZE", 1))
+
+    if run_distribute:
+        rank_size = get_group_size()
+        rank_id = get_rank()
+    else:
+        rank_size = 1
+        rank_id = 0
+    return rank_size, rank_id
+
+
+class Get_Data():
+    """
+    The data is preprocessed before being converted to midnspore.
+    """
+    def __init__(self, data_path, snr=None, training=True):
+        self.data_path = data_path + 'RML2016.10a_dict.pkl'
+        self.snr = snr
+        self.do_train = training
+        self.data_file = open(self.data_path, 'rb')
+        self.all_data = pickle.load(self.data_file, encoding='iso-8859-1')
+        self.snrs, self.mods = map(lambda j: sorted(list(set(map(lambda x: x[j], self.all_data.keys())))), [1, 0])
+
+    def to_one_hot(self, label_index):
+        """generate one hot label"""
+        one_hot_label = np.zeros([len(label_index), max(label_index) + 1])
+        one_hot_label[np.arange(len(label_index)), label_index] = 1
+        return one_hot_label
+
+    def get_loader(self):
+        train_data = []
+        train_label = []
+
+        test_data = []
+        test_label = []
+
+        train_index = []
+
+        for key in self.all_data.keys():
+            v = self.all_data[key]
+            train_data.append(v[:v.shape[0]//2])
+            test_data.append(v[v.shape[0]//2:])
+            for i in range(self.all_data[key].shape[0]//2):
+                train_label.append(key)
+                test_label.append(key)
+
+        train_data = np.vstack(train_data)
+        test_data = dict(zip(self.all_data.keys(), test_data))
+
+        for i in range(0, 110000):
+            train_index.append(i)
+
+        ds_label = []
+        if self.do_train:
+            ds = train_data
+            ds = np.expand_dims(ds, axis=1)
+            ds_label = self.to_one_hot(list(map(lambda x: self.mods.index(train_label[x][0]), train_index)))
+        else:
+            ds = []
+            for mod in self.mods:
+                ds.append((test_data[(mod, self.snr)]))
+                ds_label += [mod] * test_data[(mod, self.snr)].shape[0]
+
+            ds = np.vstack(ds)
+            ds = np.expand_dims(ds, axis=1)
+            ds_label = self.to_one_hot(list(self.mods.index(x) for x in ds_label))
+
+        return ds, ds_label
+
+
+def create_dataset(data_path,
+                   batch_size=1,
+                   training=True,
+                   snr=None,
+                   target="Ascend",
+                   run_distribute=False):
+    """create dataset for train or eval"""
+    if target == "Ascend":
+        device_num, rank_id = _get_rank_info(run_distribute)
+    if training:
+        getter = Get_Data(data_path=data_path, snr=None, training=training)
+        data = getter.get_loader()
+    else:
+        getter = Get_Data(data_path=data_path, snr=snr, training=training)
+        data = getter.get_loader()
+
+    dataset_column_names = ["data", "label"]
+    if target != "Ascend" or device_num == 1:
+        if training:
+            ds = de.NumpySlicesDataset(data=data,
+                                       column_names=dataset_column_names,
+                                       shuffle=True)
+        else:
+            ds = de.NumpySlicesDataset(data=data,
+                                       column_names=dataset_column_names,
+                                       shuffle=False)
+
+    else:
+        if training:
+            ds = de.NumpySlicesDataset(data=data,
+                                       column_names=dataset_column_names,
+                                       shuffle=True,
+                                       num_shards=device_num,
+                                       shard_id=rank_id)
+        else:
+            ds = de.NumpySlicesDataset(data=data,
+                                       column_names=dataset_column_names,
+                                       shuffle=False,
+                                       num_shards=device_num,
+                                       shard_id=rank_id)
+    ds_label = [
+        C.TypeCast(mstype.float32)
+    ]
+    ds = ds.map(operations=ds_label, input_columns=["label"])
+    ds = ds.batch(batch_size, drop_remainder=True)
+
+    return ds
+
+
+if __name__ == '__main__':
+    data_getter = Get_Data(data_path='../data/', snr=2, training=False)
+    ms_ds = data_getter.get_loader()
diff --git a/research/cv/CBAM/src/get_lr.py b/research/cv/CBAM/src/get_lr.py
new file mode 100644
index 0000000000000000000000000000000000000000..54c54edd3580083684c20abb0097fd418ea3c6ff
--- /dev/null
+++ b/research/cv/CBAM/src/get_lr.py
@@ -0,0 +1,32 @@
+# Copyright 2022 Huawei Technologies Co., Ltd
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+# http://www.apache.org/licenses/LICENSE-2.0
+#
+# Unless required by applicable law or agreed to in writing, software
+# distributed under the License is distributed on an "AS IS" BASIS,
+# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
+# See the License for the specific language governing permissions and
+# limitations under the License.
+# ============================================================================
+
+"""get learning rate"""
+
+import numpy as np
+
+
+def get_lr(init_lr, total_epoch, step_per_epoch, anneal_step=250):
+    """warmup lr schedule"""
+    total_step = total_epoch * step_per_epoch
+    lr_step = []
+
+    for step in range(total_step):
+        lambda_lr = anneal_step ** 0.5 * \
+                    min((step + 1) * anneal_step ** -1.5, (step + 1) ** -0.5)
+        lr_step.append(init_lr * lambda_lr)
+    learning_rate = np.array(lr_step).astype(np.float32)
+
+    return learning_rate
diff --git a/research/cv/CBAM/src/model.py b/research/cv/CBAM/src/model.py
new file mode 100644
index 0000000000000000000000000000000000000000..970866c87280018a6d2a7cdbcf7cc60f2ef9aef1
--- /dev/null
+++ b/research/cv/CBAM/src/model.py
@@ -0,0 +1,207 @@
+# Copyright 2022 Huawei Technologies Co., Ltd
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+# http://www.apache.org/licenses/LICENSE-2.0
+#
+# Unless required by applicable law or agreed to in writing, software
+# distributed under the License is distributed on an "AS IS" BASIS,
+# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
+# See the License for the specific language governing permissions and
+# limitations under the License.
+# ============================================================================
+
+"""Generate network."""
+
+import math
+
+from mindspore import nn
+from mindspore import ops as P
+import mindspore.common.initializer as weight_init
+
+
+class ChannelAttention(nn.Cell):
+    """
+    ChannelAttention: Since each channel of the feature map is considered as a feature detector, it is meaningful
+    for the channel to focus on the "what" of a given input image;In order to effectively calculate channel attention,
+    the method of compressing the spatial dimension of input feature mapping is adopted.
+    """
+    def __init__(self, in_channel):
+        super(ChannelAttention, self).__init__()
+        self.avg_pool = P.ReduceMean(keep_dims=True)
+        self.max_pool = P.ReduceMax(keep_dims=True)
+        self.fc = nn.SequentialCell(nn.Conv2d(in_channels=in_channel, out_channels=in_channel // 16, kernel_size=1,
+                                              has_bias=False),
+                                    nn.ReLU(),
+                                    nn.Conv2d(in_channels=in_channel // 16, out_channels=in_channel, kernel_size=1,
+                                              has_bias=False))
+        self.sigmoid = nn.Sigmoid()
+
+    def construct(self, x):
+        avg_out = self.avg_pool(x, -1)
+        avg_out = self.fc(avg_out)
+        max_out = self.max_pool(x, -1)
+        max_out = self.fc(max_out)
+        out = avg_out + max_out
+
+        return self.sigmoid(out)
+
+
+class SpatialAttention(nn.Cell):
+    """
+    SpatialAttention: Different from the channel attention module, the spatial attention module focuses on the
+    "where" of the information part as a supplement to the channel attention module.
+    """
+    def __init__(self, kernel_size=7):
+        super(SpatialAttention, self).__init__()
+
+        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, pad_mode='pad', has_bias=False)
+        self.concat = P.Concat(axis=1)
+        self.sigmod = nn.Sigmoid()
+        self.reduce_mean = P.ReduceMean(keep_dims=True)
+        self.max_pool = P.ReduceMax(keep_dims=True)
+
+    def construct(self, x):
+        avg_out = self.reduce_mean(x, 1)
+        max_out = self.max_pool(x, 1)
+        x = self.concat((avg_out, max_out))
+        x = self.conv1(x)
+
+        return self.sigmod(x)
+
+
+def conv3x3(in_channels, out_channels, stride=1):
+    """
+    3x3 convolution with padding.
+    """
+    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
+                     padding=1, has_bias=False)
+
+
+class Bottleneck(nn.Cell):
+    """
+    Residual structure.
+    """
+    expansion = 4
+
+    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
+        super(Bottleneck, self).__init__()
+        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, has_bias=False)
+        self.bn1 = nn.BatchNorm2d(out_channels)
+        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
+                               pad_mode='pad', padding=1, has_bias=False)
+        self.bn2 = nn.BatchNorm2d(out_channels)
+        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, has_bias=False)
+        self.bn3 = nn.BatchNorm2d(out_channels * 4)
+        self.relu = nn.ReLU()
+
+        self.ca = ChannelAttention(out_channels * 4)
+        self.sa = SpatialAttention()
+
+        self.dowmsample = downsample
+        self.stride = stride
+
+    def construct(self, x):
+        residual = x
+        out = self.conv1(x)
+        out = self.bn1(out)
+        out = self.relu(out)
+        out = self.conv2(out)
+        out = self.bn2(out)
+        out = self.relu(out)
+        out = self.conv3(out)
+        out = self.bn3(out)
+        out = self.ca(out) * out
+        out = self.sa(out) * out
+
+        if self.dowmsample is not None:
+            residual = self.dowmsample(x)
+
+        out += residual
+        out = self.relu(out)
+
+        return out
+
+
+class ResNet(nn.Cell):
+    """
+    Overall network architecture.
+    """
+    def __init__(self, block, layers, num_classes=11, phase="train"):
+        self.in_channels = 64
+        super(ResNet, self).__init__()
+        self.phase = phase
+        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, pad_mode='pad', padding=3, has_bias=False)
+        self.bn1 = nn.BatchNorm2d(64)
+        self.relu = nn.ReLU()
+        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
+        self.layer1 = self._make_layer(block, 8, layers[0])
+        self.layer2 = self._make_layer(block, 16, layers[1], stride=2)
+        self.layer3 = self._make_layer(block, 32, layers[2], stride=2)
+        self.layer4 = self._make_layer(block, 64, layers[3], stride=2)
+        self.avgpool = P.ReduceMean(keep_dims=True)
+        self.flatten = nn.Flatten()
+        self.Linear = nn.Dense(64 * block.expansion, num_classes)
+        dropout_ratio = 0.5
+        self.dropout = nn.Dropout(dropout_ratio)
+        self.softmax = nn.Softmax()
+        self.print = P.Print()
+
+    def construct(self, x):
+        x = self.conv1(x)
+        x = self.bn1(x)
+        x = self.relu(x)
+        x = self.maxpool(x)
+        x = self.layer1(x)
+        x = self.layer2(x)
+        x = self.layer3(x)
+        x = self.layer4(x)
+        x = self.avgpool(x, 3)
+        x = x.view(32, 256)
+        x = self.Linear(x)
+        x = self.softmax(x)
+
+        return x
+
+    def _make_layer(self, block, out_channels, blocks, stride=1):
+        downsample = None
+        if stride != 1 or self.in_channels != out_channels * block.expansion:
+            downsample = nn.SequentialCell(
+                nn.Conv2d(self.in_channels, out_channels * block.expansion,
+                          kernel_size=1, stride=stride, has_bias=False),
+                nn.BatchNorm2d(out_channels * block.expansion)
+            )
+        layers = []
+        layers.append(block(self.in_channels, out_channels, stride, downsample))
+        self.in_channels = out_channels * block.expansion
+        for i in range(1, blocks):
+            i += 1
+            layers.append(block(self.in_channels, out_channels))
+
+        return nn.SequentialCell(*layers)
+
+    def custom_init_weight(self):
+        """
+        Init the weight of Conv2d and Batchnorm2D in the net.
+        :return:
+        """
+        for _, cell in self.cells_and_names():
+            if isinstance(cell, nn.Conv2d):
+                n = cell.kernel_size[0] * cell.kernel_size[1] * cell.out_channels
+                cell.weight.set_data(weight_init.initializer(weight_init.Normal(sigma=math.sqrt(2. / n), mean=0),
+                                                             cell.weight.shape,
+                                                             cell.weight.dtype))
+            elif isinstance(cell, nn.BatchNorm2d):
+                cell.weight.set_data(weight_init.initializer(weight_init.One(),
+                                                             cell.weight.shape,
+                                                             cell.weight.dtype))
+
+
+def resnet50_cbam(phase="train", **kwargs):
+    """
+    Constructs a ResNet-50 model.
+    """
+    model = ResNet(Bottleneck, [3, 4, 6, 3], phase=phase, **kwargs)
+    return model
diff --git a/research/cv/CBAM/src/model_utils/config.py b/research/cv/CBAM/src/model_utils/config.py
new file mode 100644
index 0000000000000000000000000000000000000000..e4861623478a7d7e1dff8b63829a20992f2bcbe8
--- /dev/null
+++ b/research/cv/CBAM/src/model_utils/config.py
@@ -0,0 +1,130 @@
+# Copyright 2022 Huawei Technologies Co., Ltd
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+# http://www.apache.org/licenses/LICENSE-2.0
+#
+# Unless required by applicable law or agreed to in writing, software
+# distributed under the License is distributed on an "AS IS" BASIS,
+# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
+# See the License for the specific language governing permissions and
+# limitations under the License.
+# ============================================================================
+
+"""Parse arguments"""
+
+import os
+import ast
+import argparse
+from pprint import pprint, pformat
+import yaml
+
+
+class Config:
+    """
+    Configuration namespace. Convert dictionary to members.
+    """
+    def __init__(self, cfg_dict):
+        for k, v in cfg_dict.items():
+            if isinstance(v, (list, tuple)):
+                setattr(self, k, [Config(x) if isinstance(x, dict) else x for x in v])
+            else:
+                setattr(self, k, Config(v) if isinstance(v, dict) else v)
+
+    def __str__(self):
+        return pformat(self.__dict__)
+
+    def __repr__(self):
+        return self.__str__()
+
+
+def parse_cli_to_yaml(parser, cfg, helper=None, choices=None, cfg_path="default_config.yaml"):
+    """
+    Parse command line arguments to the configuration according to the default yaml.
+
+    Args:
+        parser: Parent parser.
+        cfg: Base configuration.
+        helper: Helper description.
+        cfg_path: Path to the default yaml config.
+    """
+    parser = argparse.ArgumentParser(description="[REPLACE THIS at config.py]",
+                                     parents=[parser])
+    helper = {} if helper is None else helper
+    choices = {} if choices is None else choices
+    for item in cfg:
+        if not isinstance(cfg[item], list) and not isinstance(cfg[item], dict):
+            help_description = helper[item] if item in helper else "Please reference to {}".format(cfg_path)
+            choice = choices[item] if item in choices else None
+            if isinstance(cfg[item], bool):
+                parser.add_argument("--" + item, type=ast.literal_eval, default=cfg[item], choices=choice,
+                                    help=help_description)
+            else:
+                parser.add_argument("--" + item, type=type(cfg[item]), default=cfg[item], choices=choice,
+                                    help=help_description)
+    args = parser.parse_args()
+    return args
+
+
+def parse_yaml(yaml_path):
+    """
+    Parse the yaml config file.
+
+    Args:
+        yaml_path: Path to the yaml config.
+    """
+    with open(yaml_path, 'r') as fin:
+        try:
+            cfgs = yaml.load_all(fin.read(), Loader=yaml.FullLoader)
+            cfgs = [x for x in cfgs]
+            if len(cfgs) == 1:
+                cfg_helper = {}
+                cfg = cfgs[0]
+                cfg_choices = {}
+            elif len(cfgs) == 2:
+                cfg, cfg_helper = cfgs
+                cfg_choices = {}
+            elif len(cfgs) == 3:
+                cfg, cfg_helper, cfg_choices = cfgs
+            else:
+                raise ValueError("At most 3 docs (config, description for help, choices) are supported in config yaml")
+            print(cfg_helper)
+        except:
+            raise ValueError("Failed to parse yaml")
+    return cfg, cfg_helper, cfg_choices
+
+
+def merge(args, cfg):
+    """
+    Merge the base config from yaml file and command line arguments.
+
+    Args:
+        args: Command line arguments.
+        cfg: Base configuration.
+    """
+    args_var = vars(args)
+    for item in args_var:
+        cfg[item] = args_var[item]
+    return cfg
+
+
+def get_config():
+    """
+    Get Config according to the yaml file and cli arguments.
+    """
+    parser = argparse.ArgumentParser(description="default name", add_help=False)
+    current_dir = os.path.dirname(os.path.abspath(__file__))
+    parser.add_argument("--config_path", type=str, default=os.path.join(current_dir, "../../default_config.yaml"),
+                        help="Config file path")
+    path_args, _ = parser.parse_known_args()
+    default, helper, choices = parse_yaml(path_args.config_path)
+    args = parse_cli_to_yaml(parser=parser, cfg=default, helper=helper, choices=choices, cfg_path=path_args.config_path)
+    final_config = merge(args, default)
+    pprint(final_config)
+    print("Please check the above information for the configurations", flush=True)
+    return Config(final_config)
+
+
+config = get_config()
diff --git a/research/cv/CBAM/src/model_utils/device_adapter.py b/research/cv/CBAM/src/model_utils/device_adapter.py
new file mode 100644
index 0000000000000000000000000000000000000000..f2fdfbde2e8b7cc98bb5dd52da273c688fab1dfe
--- /dev/null
+++ b/research/cv/CBAM/src/model_utils/device_adapter.py
@@ -0,0 +1,27 @@
+# Copyright 2022 Huawei Technologies Co., Ltd
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+# http://www.apache.org/licenses/LICENSE-2.0
+#
+# Unless required by applicable law or agreed to in writing, software
+# distributed under the License is distributed on an "AS IS" BASIS,
+# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
+# See the License for the specific language governing permissions and
+# limitations under the License.
+# ============================================================================
+
+"""Device adapter for ModelArts"""
+
+from src.model_utils.config import config
+
+if config.enable_modelarts:
+    from src.model_utils.moxing_adapter import get_device_id, get_device_num, get_rank_id, get_job_id
+else:
+    from src.model_utils.local_adapter import get_device_id, get_device_num, get_rank_id, get_job_id
+
+__all__ = [
+    "get_device_id", "get_device_num", "get_rank_id", "get_job_id"
+]
diff --git a/research/cv/CBAM/src/model_utils/local_adapter.py b/research/cv/CBAM/src/model_utils/local_adapter.py
new file mode 100644
index 0000000000000000000000000000000000000000..6b8285ca81a6010e5d69cb052059071c1c7d120d
--- /dev/null
+++ b/research/cv/CBAM/src/model_utils/local_adapter.py
@@ -0,0 +1,37 @@
+# Copyright 2022 Huawei Technologies Co., Ltd
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+# http://www.apache.org/licenses/LICENSE-2.0
+#
+# Unless required by applicable law or agreed to in writing, software
+# distributed under the License is distributed on an "AS IS" BASIS,
+# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
+# See the License for the specific language governing permissions and
+# limitations under the License.
+# ============================================================================
+
+"""Local adapter"""
+
+import os
+
+
+def get_device_id():
+    device_id = os.getenv('DEVICE_ID', '0')
+    return int(device_id)
+
+
+def get_device_num():
+    device_num = os.getenv('RANK_SIZE', '1')
+    return int(device_num)
+
+
+def get_rank_id():
+    global_rank_id = os.getenv('RANK_ID', '0')
+    return int(global_rank_id)
+
+
+def get_job_id():
+    return "Local Job"
diff --git a/research/cv/CBAM/src/model_utils/moxing_adapter.py b/research/cv/CBAM/src/model_utils/moxing_adapter.py
new file mode 100644
index 0000000000000000000000000000000000000000..a5a5876c004d27fd2f7f38272ae56a1f63c5dcc4
--- /dev/null
+++ b/research/cv/CBAM/src/model_utils/moxing_adapter.py
@@ -0,0 +1,124 @@
+# Copyright 2022 Huawei Technologies Co., Ltd
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+# http://www.apache.org/licenses/LICENSE-2.0
+#
+# Unless required by applicable law or agreed to in writing, software
+# distributed under the License is distributed on an "AS IS" BASIS,
+# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
+# See the License for the specific language governing permissions and
+# limitations under the License.
+# ============================================================================
+
+"""Moxing adapter for ModelArts"""
+
+import os
+import functools
+from mindspore import context
+from mindspore.profiler import Profiler
+from src.model_utils.config import config
+
+_global_sync_count = 0
+
+
+def get_device_id():
+    device_id = os.getenv('DEVICE_ID', '0')
+    return int(device_id)
+
+
+def get_device_num():
+    device_num = os.getenv('RANK_SIZE', '1')
+    return int(device_num)
+
+
+def get_rank_id():
+    global_rank_id = os.getenv('RANK_ID', '0')
+    return int(global_rank_id)
+
+
+def get_job_id():
+    job_id = os.getenv('JOB_ID')
+    job_id = job_id if job_id != "" else "default"
+    return job_id
+
+
+def sync_data(from_path, to_path):
+    """
+    Download data from remote obs to local directory if the first url is remote url and the second one is local path
+    Upload data from local directory to remote obs in contrast.
+    """
+    import moxing as mox
+    import time
+    global _global_sync_count
+    sync_lock = "/tmp/copy_sync.lock" + str(_global_sync_count)
+    _global_sync_count += 1
+
+    # Each server contains 8 devices as most.
+    if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
+        print("from path: ", from_path)
+        print("to path: ", to_path)
+        mox.file.copy_parallel(from_path, to_path)
+        print("===finish data synchronization===")
+        try:
+            os.mknod(sync_lock)
+        except IOError:
+            pass
+        print("===save flag===")
+
+    while True:
+        if os.path.exists(sync_lock):
+            break
+        time.sleep(1)
+
+    print("Finish sync data from {} to {}.".format(from_path, to_path))
+
+
+def moxing_wrapper(pre_process=None, post_process=None):
+    """
+    Moxing wrapper to download dataset and upload outputs.
+    """
+    def wrapper(run_func):
+        @functools.wraps(run_func)
+        def wrapped_func(*args, **kwargs):
+            # Download data from data_url
+            if config.enable_modelarts:
+                if config.data_url:
+                    sync_data(config.data_url, config.data_path)
+                    print("Dataset downloaded: ", os.listdir(config.data_path))
+                if config.checkpoint_url:
+                    sync_data(config.checkpoint_url, config.load_path)
+                    print("Preload downloaded: ", os.listdir(config.load_path))
+                if config.train_url:
+                    sync_data(config.train_url, config.output_path)
+                    print("Workspace downloaded: ", os.listdir(config.output_path))
+
+                context.set_context(save_graphs_path=os.path.join(config.output_path, str(get_rank_id())))
+                config.device_num = get_device_num()
+                config.device_id = get_device_id()
+                if not os.path.exists(config.output_path):
+                    os.makedirs(config.output_path)
+
+                if pre_process:
+                    pre_process()
+
+            if config.enable_profiling:
+                profiler = Profiler()
+
+            run_func(*args, **kwargs)
+
+            if config.enable_profiling:
+                profiler.analyse()
+
+            # Upload data to train_url
+            if config.enable_modelarts:
+                if post_process:
+                    post_process()
+
+                if config.train_url:
+                    print("Start to copy output directory")
+                    sync_data(config.output_path, config.train_url)
+        return wrapped_func
+    return wrapper
diff --git a/research/cv/CBAM/train.py b/research/cv/CBAM/train.py
new file mode 100644
index 0000000000000000000000000000000000000000..2ce1eea9b2817b0710dc1f202f6e135ac8a81793
--- /dev/null
+++ b/research/cv/CBAM/train.py
@@ -0,0 +1,115 @@
+# Copyright 2022 Huawei Technologies Co., Ltd
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+# http://www.apache.org/licenses/LICENSE-2.0
+#
+# Unless required by applicable law or agreed to in writing, software
+# distributed under the License is distributed on an "AS IS" BASIS,
+# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
+# See the License for the specific language governing permissions and
+# limitations under the License.
+# ============================================================================
+
+"""
+##############train CBAM example on dataset#################
+python train.py
+"""
+
+from mindspore.common import set_seed
+from mindspore import context
+from mindspore.communication.management import init
+from mindspore.context import ParallelMode
+from mindspore.nn.optim import Adam
+from mindspore.train.model import Model
+from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
+from mindspore.nn.metrics import Accuracy
+from mindspore.nn import BCEWithLogitsLoss
+
+
+from src.model_utils.config import config
+from src.model_utils.moxing_adapter import moxing_wrapper
+from src.model_utils.device_adapter import get_device_id, get_device_num, get_rank_id, get_job_id
+from src.data import create_dataset
+from src.model import resnet50_cbam
+from src.get_lr import get_lr
+
+set_seed(1)
+
+
+def modelarts_pre_process():
+    pass
+
+
+@moxing_wrapper(pre_process=modelarts_pre_process)
+def run_train():
+    """train function"""
+    print('device id:', get_device_id())
+    print('device num:', get_device_num())
+    print('rank id:', get_rank_id())
+    print('job id:', get_job_id())
+
+    device_target = config.device_target
+    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
+    context.set_context(save_graphs=False)
+    if config.device_target == "GPU":
+        context.set_context(enable_graph_kernel=True)
+        context.set_context(graph_kernel_flags="--enable_cluster_ops=MatMul")
+
+    device_num = get_device_num()
+
+    if config.run_distribute:
+        context.reset_auto_parallel_context()
+        context.set_auto_parallel_context(device_num=device_num,
+                                          parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
+        if device_target == "Ascend":
+            context.set_context(device_id=get_device_id())
+            init()
+        elif device_target == "GPU":
+            init()
+    else:
+        context.set_context(device_id=get_device_id())
+    print("init finished.")
+
+    ds_train = create_dataset(data_path=config.data_path, batch_size=config.batch_size, training=True,
+                              snr=None, target=config.device_target, run_distribute=config.run_distribute)
+    if ds_train.get_dataset_size() == 0:
+        raise ValueError("Please check dataset size > 0 and batch_size <= dataset size.")
+    print("create dataset finished.")
+    step_per_size = ds_train.get_dataset_size()
+    print("train dataset size:", step_per_size)
+
+    net = resnet50_cbam(phase="train")
+    loss = BCEWithLogitsLoss()
+    lr = get_lr(0.001, config.epoch_size, step_per_size, step_per_size * 2)
+    opt = Adam(net.trainable_params(),
+               learning_rate=lr,
+               beta1=0.9,
+               beta2=0.999,
+               eps=1e-7,
+               weight_decay=0.0,
+               loss_scale=1.0)
+
+    metrics = {"Accuracy": Accuracy()}
+    model = Model(net, loss_fn=loss, optimizer=opt, metrics=metrics)
+
+    time_cb = TimeMonitor()
+    loss_cb = LossMonitor(per_print_times=step_per_size)
+    callbacks_list = [time_cb, loss_cb]
+    config_ck = CheckpointConfig(save_checkpoint_steps=step_per_size * 10,
+                                 keep_checkpoint_max=100)
+    if get_rank_id() == 0:
+        ckpoint_cb = ModelCheckpoint(prefix='cbam_train', directory=config.ckpt_path, config=config_ck)
+        callbacks_list.append(ckpoint_cb)
+
+    print("train start!")
+    model.train(epoch=config.epoch_size,
+                train_dataset=ds_train,
+                callbacks=callbacks_list,
+                dataset_sink_mode=config.dataset_sink_mode)
+
+
+if __name__ == '__main__':
+    run_train()
