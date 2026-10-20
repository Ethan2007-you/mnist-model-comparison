# MNIST 模型比较

这是一个用于学习机器学习分类流程的入门项目。项目使用 PyTorch 全连接神经网络和 scikit-learn 线性 SVM 识别 MNIST 手写数字。神经网络部分比较不同 Dropout 比例；SVM 部分记录验证集、测试集准确率、训练时间和测试集混淆矩阵。后续可以继续加入随机森林。

## 当前功能

- 下载并加载 MNIST 数据集
- 将官方训练集拆分为训练集（50,000 张）和验证集（10,000 张）
- 比较 Dropout 比例 `0.0`、`0.2` 和 `0.5`
- 使用 `LinearSVC` 作为线性核 SVM 基线
- 根据验证集准确率选择模型，再对测试集进行一次最终评估
- 将神经网络每轮训练记录保存到 `results/dropout_comparison.csv`
- 将 SVM 调参记录保存到 `results/svm_tuning.csv`，最终指标保存到 `results/svm_metrics.csv`，测试集混淆矩阵保存到 `results/svm_confusion_matrix.csv`
- 自动使用 CUDA（可用时），否则使用 CPU

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
│   └── train_svm.py
├── data/       # 运行时下载，不提交到 Git
└── results/    # 训练记录
```

## 实验注意事项

- 训练集用于更新模型参数，验证集用于选择 Dropout 配置，测试集用于最终评估。
- 不要根据测试集结果反复调参，否则测试集就失去了独立评估的作用。
- Dropout 只在训练模式下随机丢弃神经元；评估时通过 `model.eval()` 关闭。
