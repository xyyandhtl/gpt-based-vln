# gpt-based-vln
记录解释性强的、综合易移植性和可调试性的基于VL-LLM的视觉语言导航工作

- 选择原则
  - 尽量不用原始端到端形态的seq-to-seq
  - 优先gpt-based，或解释性强的另一种形态，实时map构建和作为认知input。gpt+map最佳。
  - zero-shot+fine-tuned > zero-shot > pretrained
  - 尽量模块化，代码解释性强
  - 三方依赖尽量少

## InstructNav

- 完全的zero-shot，不需要任何vln相关预训练
- 安装detectron2需源码编译，torch需安装一定对应cuda版本的，加后缀 pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
- 使用glee。
- 仅包含objnav任务源码，issue里作者24年10月提到vln-ce源码也将开源

## PixelNavigator

- 半zero-shot，需要一个PixelNav的类人skill的预训练决策器，但因为把场景化的因素大部分剥离到vl-llm处理了，通用性理论上大大增加
- PixelNav训练
- grounding-dino+sam使用源码安装，考虑替换为huggingface的版本。sam2已出，考虑替换。
- 和glee相比在实例分割上上限谁高？glee作为面向对象实例的通用框架，还支持视频、提示、追踪等应用，可拓展性更高
- 同样仅包含objnav任务源码

## TagMap

- todo

## GridMM (no gpt-based, map-based)

- 虽还是seq-to-seq，但维护了grid map，有一定解释性，并且地图特征的encoder处理进一步增加了解释性
- 12900K+3090预训练～70h, fine-tune～48h

## BevBert (no gpt-based, map-based)

- 拓扑图和度量图（Topo-Metric）思想借鉴
- 对于地图的特征input的transformer encoder感觉不如GridMM精致，可解释性差点

## Prospect
[notations.md](overview/notations.md)
