diff --git a/.gitee/PULL_REQUEST_TEMPLATE.md b/.gitee/PULL_REQUEST_TEMPLATE.md
index 461682c0555ccb909c304021259918c2fc5b915c..cb027d4b2a7a877eb84b09d9da04948b8fde57e2 100644
--- a/.gitee/PULL_REQUEST_TEMPLATE.md
+++ b/.gitee/PULL_REQUEST_TEMPLATE.md
@@ -36,7 +36,7 @@ Fixes #
 - [ ] I have made corresponding changes to the documentation.
 - [ ] I have squashed all the commits into one.
 - [ ] I have test and ascertained the effect of my change in all related cases.
-    - [ ] Different hardware: `CPU`, `GPU`, `Ascend910`, `Ascend310`, `Ascend701`.
+    - [ ] Different hardware: `CPU`, `GPU`, `Ascend910`, `Ascend310`, `Ascend310P`.
     - [ ] Different mode: `GRAPH_MODE`, `PYNATIVE_MODE`.
     - [ ] Different system: `Linux`, `Windows`, `MAC`.
     - [ ] Different number of cluster: `1pc`, `8pcs`.
diff --git a/how_to_contribute/README_TEMPLATE.md b/how_to_contribute/README_TEMPLATE.md
index 730463747bd4f0cabd9219eb53cb9afe09db1f04..7e5953866732683d79353c4af0c2fe4adc26f679 100644
--- a/how_to_contribute/README_TEMPLATE.md
+++ b/how_to_contribute/README_TEMPLATE.md
@@ -1,6 +1,6 @@
-<TOC>
+# Content
 
-# Title, Model name
+# Model name
 
 > The Description of Model. The paper present this model.
 
@@ -8,13 +8,13 @@
 
 > There could be various architecture about some model. Represent the architecture of your implementation.
 
-## Features(optional)
+## Dataset
 
-> Represent the distinctive feature you used in the model implementation. Such as distributed auto-parallel or some special training trick.
+> Provide the information of the dataset you used. Check the copyrights of the dataset you used, usually you need to provide the hyperlink to download the dataset, scope and data size.
 
-## Dataset
+## Features(optional)
 
-> Provide the information of the dataset you used. Check the copyrights of the dataset you used, usually you need to provide the hyperlink to download the dataset.
+> Represent the distinctive feature you used in the model implementation. Such as distributed auto-parallel or some special training trick.
 
 ## Requirements
 
@@ -28,6 +28,10 @@
 ## Quick Start
 
 > How to take a try without understanding anything about the model.
+> Maybe include：
+> * run train，run eval，run export
+> * Ascend version, GPU version，CPU version
+> * offline version，ModelArts version
 
 ## Script Description
 
@@ -35,15 +39,15 @@
 
 ### Scripts and Sample Code
 
-> Explain every file in your project.
+> Show the scope of project(include children directory), Explain every file in your project.
 
 ### Script Parameter
 
-> Explain every parameter of the model. Especially the parameters in `config.py`.
+> Explain every parameter of the model. Especially the parameters in `config.py`. If there are multiple config files, please explain separately.
 
 ## Training
 
-> Provide training information.
+> Provide training information. Include usage and log.
 
 ### Training Process
 
@@ -55,28 +59,57 @@ e.g. Run the following command for distributed training on Ascend.
 bash run_distribute_train.sh [RANK_TABLE_FILE] [PRETRAINED_MODEL]
 ```
 
-### Transfer Training(Optional)
-
-> Provide the guidelines about how to run transfer training based on an pretrained model.
-
-### Training Result
+> Provide training logs.
 
-> Provide the result of training.
+```log
+# grep "loss is " train.log
+epoch:1 step:390, loss is 1.4842823
+epcoh:2 step:390, loss is 1.0897788
+```
 
+> Provide training result.
 e.g. Training checkpoint will be stored in `XXXX/ckpt_0`. You will get result from log file like the following:
 
-```
+```log
 epoch: 11 step: 7393 ,rpn_loss: 0.02003, rcnn_loss: 0.52051, rpn_cls_loss: 0.01761, rpn_reg_loss: 0.00241, rcnn_cls_loss: 0.16028, rcnn_reg_loss: 0.08411, rcnn_mask_loss: 0.27588, total_loss: 0.54054
 epoch: 12 step: 7393 ,rpn_loss: 0.00547, rcnn_loss: 0.39258, rpn_cls_loss: 0.00285, rpn_reg_loss: 0.00262, rcnn_cls_loss: 0.08002, rcnn_reg_loss: 0.04990, rcnn_mask_loss: 0.26245, total_loss: 0.39804
 ```
 
+### Transfer Training(Optional)
+
+> Provide the guidelines about how to run transfer training based on an pretrained model.
+
+### Distribute Training
+
+> Same as Training
+
 ## Evaluation
 
-### Evaluation Process
+### Evaluation Process 910
 
 > Provide the use of evaluation scripts.
 
-### Evaluation Result
+### Evaluation Result 910
+
+> Provide the result of evaluation.
+
+## Export
+
+### Export Process
+
+> Provide the use of export scripts.
+
+### Export Result
+
+> Provide the result of export.
+
+## Evaluation 310
+
+### Evaluation Process 310
+
+> Provide the use of evaluation scripts.
+
+### Evaluation Result 310
 
 > Provide the result of evaluation.
 
@@ -134,10 +167,13 @@ e.g. you can reference the following template
 
 ## Contributions
 
+This part should not exist in your readme.
 If you want to contribute, please review the [contribution guidelines](https://gitee.com/mindspore/models/blob/master/CONTRIBUTING.md) and [how_to_contribute](https://gitee.com/mindspore/models/tree/master/how_to_contribute)
 
 ### Contributors
 
+Update your school and email/gitee.
+
 * [c34](https://gitee.com/c_34) (Huawei)
 
 ## ModeZoo Homepage
diff --git a/how_to_contribute/README_TEMPLATE_CN.md b/how_to_contribute/README_TEMPLATE_CN.md
index 3d160f1e32f5d59e49e3baf7504f1e6bf109b553..ae5daac27423ea13c2435a242154679f518c31dd 100644
--- a/how_to_contribute/README_TEMPLATE_CN.md
+++ b/how_to_contribute/README_TEMPLATE_CN.md
@@ -1,16 +1,20 @@
-<TOC>
+# 目录
 
-# 标题， 模型名称
+# 模型名称
 
-> 可以是模型的不同架构，名称可以代表你所实现的模型架构
+> 模型简介，论文模型概括
 
-## 特性（可选）
+## 模型架构
 
-> 展示你在模型实现中使用的特性，例如分布式自动并行或者一些特殊的训练技巧
+> 如包含多种模型架构，展示你实现的部分
 
 ## 数据集
 
-> 提供你所使用的数据信息，检查数据版权，通常情况下你需要提供下载数据的链接
+> 提供你所使用的数据信息，检查数据版权，通常情况下你需要提供下载数据的链接，数据集的目录结构，数据集大小等信息
+
+## 特性（可选）
+
+> 展示你在模型实现中使用的特性，例如分布式自动并行或者混合精度等一些特殊的训练技巧
 
 ## 环境要求
 
@@ -23,7 +27,11 @@
 
 ## 快速入门
 
-> 使用一条什么样的命令可以直接运行
+> 展示可以直接运行的命令
+> 按照你开发的版本，可能包含：
+> * 训练命令，推理命令，export命令
+> * Ascend版本，GPU版本，CPU版本
+> * 线下运行版本，线上运行版本
 
 ## 脚本说明
 
@@ -31,19 +39,19 @@
 
 ### 脚本和样例代码
 
-> 描述项目中每个文件的作用
+> 提供完整的代码目录展示（包含子文件夹的展开），描述每个文件的作用
 
 ### 脚本参数
 
-> 注释模型中的每个参数，特别是`config.py`中的参数
+> 注解模型中的每个参数，特别是`config.py`中的参数，如有多个配置文件，请注解每一份配置文件的参数
 
 ## 训练过程
 
-> 提供训练信息
+> 提供训练信息，区别于quick start，此部分需要提供除用法外的日志等详细信息
 
-### 用法
+### 训练
 
-> 提供训练脚本的使用情况
+> 提供训练脚本的使用方法
 
 例如：在昇腾上使用分布式训练运行下面的命令
 
@@ -51,20 +59,55 @@
 bash run_distribute_train.sh [RANK_TABLE_FILE] [PRETRAINED_MODEL]
 ```
 
+> 提供训练过程日志
+
+```log
+# grep "loss is " train.log
+epoch:1 step:390, loss is 1.4842823
+epcoh:2 step:390, loss is 1.0897788
+```
+
+> 提供训练结果日志
+例如：训练checkpoint将被保存在`XXXX/ckpt_0`中，你可以从如下的log文件中获取结果
+
+```log
+epoch: 11 step: 7393 ,rpn_loss: 0.02003, rcnn_loss: 0.52051, rpn_cls_loss: 0.01761, rpn_reg_loss: 0.00241, rcnn_cls_loss: 0.16028, rcnn_reg_loss: 0.08411, rcnn_mask_loss: 0.27588, total_loss: 0.54054
+epoch: 12 step: 7393 ,rpn_loss: 0.00547, rcnn_loss: 0.39258, rpn_cls_loss: 0.00285, rpn_reg_loss: 0.00262, rcnn_cls_loss: 0.08002, rcnn_reg_loss: 0.04990, rcnn_mask_loss: 0.26245, total_loss: 0.39804
+```
+
 ### 迁移训练（可选）
 
 > 提供如何根据预训练模型进行迁移训练的指南
 
-### 训练结果
+### 分布式训练
 
-> 提供训练结果
+> 同上
 
-例如：训练checkpoint将被保存在`XXXX/ckpt_0`中，你可以从如下的log文件中获取结果
+## 评估
 
+### 评估过程
+
+> 提供eval脚本用法
+
+### 评估结果
+
+> 提供推理结果
+
+例如：上述python命令将在后台运行，您可以通过eval.log文件查看结果。测试数据集的准确性如下：
+
+```log
+accuracy:{'acc':0.934}
 ```
-epoch: 11 step: 7393 ,rpn_loss: 0.02003, rcnn_loss: 0.52051, rpn_cls_loss: 0.01761, rpn_reg_loss: 0.00241, rcnn_cls_loss: 0.16028, rcnn_reg_loss: 0.08411, rcnn_mask_loss: 0.27588, total_loss: 0.54054
-epoch: 12 step: 7393 ,rpn_loss: 0.00547, rcnn_loss: 0.39258, rpn_cls_loss: 0.00285, rpn_reg_loss: 0.00262, rcnn_cls_loss: 0.08002, rcnn_reg_loss: 0.04990, rcnn_mask_loss: 0.26245, total_loss: 0.39804
-```
+
+## 导出
+
+### 导出过程
+
+> 提供export脚本用法
+
+### 导出结果
+
+> 提供export结果日志
 
 ## 推理
 
@@ -72,6 +115,10 @@ epoch: 12 step: 7393 ,rpn_loss: 0.00547, rcnn_loss: 0.39258, rpn_cls_loss: 0.002
 
 > 提供推理脚本
 
+```bash
+bash run_infer_310.sh [MINDIR_PATH] [DATASET] [DATA_PATH] [LABEL_FILE] [DEVICE_ID]
+```
+
 ### 推理结果
 
 > 提供推理结果
@@ -126,6 +173,7 @@ epoch: 12 step: 7393 ,rpn_loss: 0.00547, rcnn_loss: 0.39258, rpn_cls_loss: 0.002
 
 ## 参考模板
 
+此部分不需要出现在你的README中
 [maskrcnn_readme](https://gitee.com/mindspore/models/blob/master/official/cv/maskrcnn/README_CN.md)
 
 ## 贡献指南
@@ -134,6 +182,8 @@ epoch: 12 step: 7393 ,rpn_loss: 0.00547, rcnn_loss: 0.39258, rpn_cls_loss: 0.002
 
 ### 贡献者
 
+此部分根据自己的情况进行更改，填写自己的院校和邮箱
+
 * [c34](https://gitee.com/c_34) (Huawei)
 
 ## ModelZoo 主页# learning
