修改了打开文件部分代码，使得可以手动选择文件路径及能够正常在Windows正常运行  

使用uv管理的虚拟环境进行测试，conda可能有bug  

添加了 --data_root 参数选择数据集  

——————————

# 模型更新

我们提出了一种 2.5D 输入策略：在推理与训练阶段以当前切片及其相邻切片构成的三联切片作为网络输入，在不引入完整 3D 卷积的前提下显式编码局部体数据上下文。不同于直接设计复杂的 3D 主干网络，我们的 2.5D 方案仅在输入端引入极小的结构改动，对原有 U‑SAM 主干完全透明，因此 几乎不增加参数量与计算开销，却显著提升了跨切片的一致性和边界连续性。在直肠等长轴方向结构连续、单切片对比度有限的场景中，该设计尤其能够缓解“孤立切片”带来的伪阳性和漏检问题。

如果直接将三联切片简单拼接后送入预训练的 SAM 编码器，会导致输入分布与自然图像预训练阶段存在明显差异，削弱预训练权重的迁移效果为此，我们在 U‑SAM 前端设计了一个 轻量级输入融合 stem：通过两层conv对 2.5D 输入进行特征融合与重投影，使其重新映射回三通道空间。该模块一方面在通道维度上聚合三张切片的上下文信息，另一方面保持与原始 SAM 输入接口完全兼容，因而 在不修改主干结构和预训练参数的前提下，自适应地完成了从 2.5D 医学图像到 SAM 输入域的对齐。

# 如何使用

默认输出尺寸为 224*224，后期将 npz 上采样到 512

## pth转onnx脚本文件[convert_pth_to_onnx](convert_pth_to_onnx.py)

```
# 基本用法
python convert_pth_to_onnx.py --pth_path /path/to/your/model.pth

# 指定输出路径
python convert_pth_to_onnx.py --pth_path /path/to/your/model.pth --output /path/to/output.onnx

# 自定义参数
python convert_pth_to_onnx.py --pth_path /path/to/your/model.pth \
    --img_size 224 \
    --sam_num_classes 3 \
    --opset_version 14
```

[这是一些小工具](Rare-tools)  
[使用说明](Start.md)  
[原始的Readme](Origin-README.md)