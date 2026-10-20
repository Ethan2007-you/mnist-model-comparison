# MNIST 模型比较

这是一个用于学习机器学习分类流程的入门项目。项目使用 PyTorch 全连接神经网络、基础 CNN、LeNet-5 风格 CNN、scikit-learn 线性 SVM 和随机森林识别 MNIST 手写数字。MLP 部分比较不同 Dropout 比例；各模型使用相同的数据划分，并记录验证集和测试集指标。

## 当前功能

- 下载并加载 MNIST 数据集
- 将官方训练集拆分为训练集（50,000 张）和验证集（10,000 张）
- 比较 Dropout 比例 `0.0`、`0.2` 和 `0.5`
- 使用基础 CNN 提取图片中的局部笔画特征
- 使用 LeNet-5 风格网络比较经典卷积架构
- 使用 `LinearSVC` 作为线性核 SVM 基线
- 使用 `RandomForestClassifier` 作为树模型基线
- 根据验证集准确率为各个模型选择配置或训练轮次，再进行最终测试集评估
- 将神经网络每轮训练记录保存到 `results/dropout_comparison.csv`
- 将 CNN 每轮训练记录、最终指标和测试集混淆矩阵分别保存到 `results/` 下的 CSV 文件
- 将 SVM 和随机森林的调参记录、最终指标及测试集混淆矩阵分别保存到 `results/` 下的 CSV 文件
- 神经网络可用时自动使用 CUDA；SVM 和随机森林使用 CPU

## 环境准备

建议使用 Python 3.10 或更新版本。在项目根目录运行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果 PowerShell 阻止虚拟环境激活，也可以直接使用虚拟环境中的 Python：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 运行

```powershell
python src/train_nn.py
```

可以调整训练轮数或要比较的 Dropout 比例：

```powershell
python src/train_nn.py --epochs 5 --dropout-rates 0.0 0.2 0.5
```

运行基础 CNN：

```powershell
python src/train_cnn.py
```

默认训练 5 轮，使用两层卷积和最大池化保留图像空间结构。可调整训练轮数、批次大小和学习率：

```powershell
python src/train_cnn.py --epochs 5 --batch-size 64 --learning-rate 0.001
```

CNN 会根据验证集准确率保留最佳轮次的权重，再对测试集进行最终评估。训练记录写入 `results/cnn_training_history.csv`，测试指标和混淆矩阵分别写入 `results/cnn_metrics.csv` 和 `results/cnn_confusion_matrix.csv`。

运行 LeNet-5 风格 CNN：

```powershell
python src/train_lenet.py
```

该实现将 28×28 图片补零到 32×32，再依次使用卷积、Tanh、平均池化和全连接层，体现 LeNet-5 的经典结构。为适配现代 PyTorch 分类训练，最后输出 10 类 logits 并使用交叉熵损失。也可以调整训练参数：

```powershell
python src/train_lenet.py --epochs 5 --batch-size 64 --learning-rate 0.001
```

训练历史、最终指标和测试集混淆矩阵分别保存到 `results/lenet_training_history.csv`、`results/lenet_metrics.csv` 和 `results/lenet_confusion_matrix.csv`。

运行线性 SVM：

```powershell
python src/train_svm.py
```

默认使用全部 50,000 张训练图片。SVM 在 CPU 上运行；如果想先快速试跑，可以减少训练图片数：

```powershell
python src/train_svm.py --max-train-samples 5000 --c-value 1.0
```

`C` 控制分类错误与模型复杂度之间的权衡。调参时可以只看验证集：

```powershell
python src/train_svm.py --c-value 0.1 --skip-test
python src/train_svm.py --c-value 1.0 --skip-test
python src/train_svm.py --c-value 10 --skip-test
```

确定 `C` 后，再运行一次不带 `--skip-test` 的命令，完成最终测试评估并生成混淆矩阵。当前实现使用线性 SVM，适合直接对 MNIST 的 784 个像素特征进行基线比较。RBF 核 SVM 在 50,000 张图片上计算开销较大，因此暂未作为默认配置。

运行随机森林：

```powershell
python src/train_random_forest.py
```

默认训练 200 棵树，并在每次分裂时从 `sqrt(784)` 个候选像素特征中选择。调参时可先跳过测试集，并把每次验证结果追加到 `results/random_forest_tuning.csv`：

```powershell
python src/train_random_forest.py --n-estimators 100 --skip-test
python src/train_random_forest.py --n-estimators 200 --skip-test
```

选定参数后，运行不带 `--skip-test` 的命令执行最终测试评估。最终指标写入 `results/random_forest_metrics.csv`，混淆矩阵写入 `results/random_forest_confusion_matrix.csv`。也可用 `--max-train-samples 5000` 先缩小训练集快速体验。

第一次运行时，程序会把 MNIST 数据下载到 `data/`。训练日志和最终测试准确率会显示在终端；每个 Dropout 配置、每轮训练的损失和准确率会写入 `results/dropout_comparison.csv`。

## 项目结构

```text
mnist-model-comparison/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── mnist_data.py
│   ├── train_nn.py
│   ├── train_cnn.py
│   ├── train_lenet.py
│   ├── train_svm.py
│   └── train_random_forest.py
├── data/       # 运行时下载，不提交到 Git
└── results/    # 训练记录
```

## 实验注意事项

- 训练集用于更新模型参数，验证集用于选择 Dropout 配置，测试集用于最终评估。
- 不要根据测试集结果反复调参，否则测试集就失去了独立评估的作用。
- Dropout 只在训练模式下随机丢弃神经元；评估时通过 `model.eval()` 关闭。
