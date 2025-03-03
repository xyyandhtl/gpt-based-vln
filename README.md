# gpt-based-vln
记录解释性强的、综合易移植性和可调试性的基于VL-LLM的视觉语言导航工作。
代码整合，gpt-api调用统一，统一python环境关键依赖库及版本整理，Tips备忘记录。

- 选择原则
  - 尽量不用端到端形态的seq-to-seq
  - 优先gpt(vllm)-based，gpt作为agent使用。
  - 考虑解释性强的另一种形态：场景理解map构建，作为认知input。即优选vl-gpt with representative_map_builder的工作
  - zero-shot+fine-tuned > zero-shot > pretrained
  - 尽量模块化，代码解释性强
  - 三方/小众库依赖尽量少

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

- 不同于上面vln输出为planner甚至controller级别的指令，只输出objnav的目标坐标，是一个不完整的navigation框架
- 预建语义地图，不是未知区域探索，但地图形态可借览
- 开放词汇的语义地图，为减少原开放词汇所必须的特征embedding内存占用大，采用数千个语义类别来替代隐式embedding，覆盖语义广，并且是多标签标注，可至少平替现sota隐式预训练VLM语义
- 使用recognize-anything++，CLIP/BLIP的上位替代

## VLMaps

- 基于CLIP，地图形态算是TagMap的下位替代，但提供了导航controller级的指令输出，较完整的先验地图VLN框架
- 一个depth-free，仅基于稀疏重建的移植改动：[spase_vln](https://github.com/xyyandhtl/sparse-vln)

## GridMM (no gpt-based, map-based)

- 虽还是seq-to-seq，但维护了grid map，有一定解释性，并且地图特征的encoder处理进一步增加了解释性
- 12900K+3090预训练～70h, fine-tune～48h

## BevBert (no gpt-based, map-based)

- 拓扑图和度量图（Topo-Metric）思想借鉴
- 对于地图的特征input的transformer encoder感觉不如GridMM精致，可解释性差点

## NavRAG（属数据增强工作，但值得借鉴）
- Demand-Driven任务的数据增强指令生成方向最新工作
- 需求导向VLN任务要求对场景的深刻认识，所以亮点是构建了地图的新形态：Hierarchical Scene Description Tree。虽然作用是生成泛化性强的数据集，但也是非常值得借鉴的地图形态，和人对场景的理解方式很接近了
- 用Gpt生成视图、视点、区域和整个场景的分层场景描述，描述了不同级别的环境语义和空间关系，有助于LLM理解3D环境并检索用于指令生成的信息。
- todo：数据集暂无法获取TeraBox is not available in current area
- 源码还在更新，先submodule引入

## 关注的待开源工作
- Mem2Ego: https://arxiv.org/pdf/2502.14254
- MapNav: https://arxiv.org/pdf/2502.13451

## Prospect
[notations.md](overview/notations.md)
