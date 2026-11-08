# TC-IQA

## 运行

先安装与显卡驱动兼容的 PyTorch 和 torchvision，再安装项目依赖：

```powershell
python -m pip install -r requirements.txt
```

修改唯一配置文件 `configs/train.yaml` 中的数据集路径，然后运行：

```powershell
python scripts/train_conditioned.py
```

当前默认：MIQD、无参考、冻结 ConvNeXt-Tiny、四分支跨层融合、Huber-only。
