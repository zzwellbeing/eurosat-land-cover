# 卫星影像地表分类

基于 EuroSAT RGB 和 PyTorch 的十类地表识别。项目包含数据清洗和划分、三组学习率实验、独立测试、单图推理和 Flask 网页。网页接受与 EuroSAT 尺度相近的 RGB 遥感小图。

项目源码仓库地址在发布后填写于 `source_repository.txt` 和此处。

## 环境安装

实测 Python 3.12、PyTorch 2.7.1、TorchVision 0.22.1、TensorBoard 2.19.0、Flask 3.1.1 和 macOS Apple Silicon CPU。默认无需 CUDA。使用 Windows 或 Linux 的 CPU 时，可为 uv 安装命令增加 `--index https://download.pytorch.org/whl/cpu`。以下命令均从项目根目录运行。

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

## 数据与训练

```bash
uv run --no-project python -m src.data
uv run --no-project python -m src.train
uv run --no-project python -m src.evaluate
```

- 数据脚本下载 Zenodo 原始 ZIP，验证 SHA-256、解码 RGB、排除异常与重复，并按固定种子进行分层划分。下载受限时可从 <https://zenodo.org/records/7711810> 下载 `EuroSAT_RGB.zip`，放到 `dataset/`，或用 `--archive /path/to/EuroSAT_RGB.zip` 指定路径。
- `dataset/raw/` 由脚本重建。CSV 包含相对路径、类别、集合和像素 SHA-256，详见 `dataset/audit.json`。保留每类三张完全独立的现场演示图，放在 `dataset/demo/`。报告样例来自 `dataset/samples/`。
- 已交付的日志和权重可以直接查看。复跑时用新路径保留实测：
  `uv run --no-project python -m src.train --runs runs_repro --output models_repro/best_model.pth`。
  对应评估命令是 `uv run --no-project python -m src.evaluate --model models_repro/best_model.pth --output results_repro`。
- 每组运行 8 个 epoch，使用同一批划分、CNN、随机种子及 AdamW，仅调整学习率。最佳权重先按验证准确率、再按验证 Loss 排序。全局最佳权重在 `models/best_model.pth`，其中包含模型状态字典、类别顺序和验证记录。测试结果保存在 `results/`。

## 单图推理与现场演示

```bash
uv run --no-project python -m src.predict "$(find dataset/demo/Forest -name '*.jpg' | head -n 1)"
uv run --no-project tensorboard --logdir runs --host 127.0.0.1 --port 6006
uv run --no-project python -m src.app
```

TensorBoard 和 Flask 各用一个终端。TensorBoard 地址为 <http://127.0.0.1:6006>，在 Scalars 或 Time Series 中查看 Loss、Accuracy、LearningRate。Flask 地址为 <http://127.0.0.1:8000>。上传 `dataset/demo/` 中的图片，点击“开始预测”，页面会显示 Top-3 概率和本机模型耗时。API 为 `GET /api/model` 和 `POST /api/predict`，上传表单字段为 `image`。已提交模型支持离线预测。

## 常见问题

- 端口被占用时，给 Flask 或 TensorBoard 改用空闲的 `--port`。
- 模型文件缺失时，从交付包恢复 `models/best_model.pth` 或重训。需重现评估时先运行数据整理脚本。
- 日志目录已有结果时，给 `--runs` 指定空目录，以免同名实验被混合。
- 接受 JPG、PNG、WebP，文件大小上限 5 MB，尺寸上限 1200 万像素。Softmax 概率取决于模型与已知类别，普通照片和不同空间分辨率的遥感影像可能被错误分类。
- 按像素散列排除了完全相同的图像。RGB 版本的图块可能在地理位置上相邻，因此测试结果只表示本数据集划分的性能。

## 来源

- EuroSAT 数据及授权说明 <https://github.com/phelber/EuroSAT>
- EuroSAT RGB 固定版本 <https://zenodo.org/records/7711810>
- Copernicus Sentinel 数据条款 <https://cds.climate.copernicus.eu/licences/ec-sentinel>
- PyTorch 训练教程 <https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html>

样例影像来自 EuroSAT RGB。Contains modified Copernicus Sentinel data (2018, EuroSAT RGB publication). 授权文件见 `dataset/LICENSE-EuroSAT.txt`。项目代码与报告由生成式 AI 辅助编写，实测日志来自实际执行的训练脚本。

## 实测结果

共整理 27,000 张图像。训练、验证、测试和演示集合分别为 18,870、4,050、4,050 和 30 张。最佳模型为 `cnn_lr003` 的第 8 个 epoch，验证准确率 86.96%。独立测试的 Top-1 准确率为 86.07%，Top-3 命中率为 98.37%，宏平均 F1 为 0.8556。逐张测试结果、混淆矩阵和误判案例在 `results/`。运行 `uv run --no-project python -m src.figures` 可从事件日志和测试输出重画图表。

交付目录中包含原作者发布的完整 RGB ZIP。即使离线也可重建原始图像目录和划分。安装依赖时仍需访问 Python 包索引或使用已有的离线缓存。

## 填写提交信息

班级与报告封面字段留待填写。视频位于 `video/`。源码发布后，把同一可访问地址写入本文件和 `source_repository.txt`，并按班级重命名外层目录、ZIP 文件及演示视频。Git 历史以交付目录之外的 Git bundle 提供，便于克隆后连接自己的空仓库并推送。公开源码仓库时，原始数据 ZIP 可继续从 Zenodo 获取。
