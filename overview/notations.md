## InstructNav
  - Value map聚合逻辑
  - Gpt prompt定制优化
  - 垂直领域微调vlm，3d空间理解
  - VL gpt和分割model(glee or lseg)用的clip text encoder的adaption
  - 中文的迁移
  - 相似语义在不同history traj条件下的apply or not，何时放弃探索更近的语义，大部分失败例都是误导航到相似语义处停止

## PixelNavigator
  - 和InstructNav同一团队，思想有点像，代码结构类同
  - 预计场景适应性弱一点，但鲁棒性强一点
  - 地图认知的植入
  - pixelnav的进一步非通用因素剥离，加入通用input
  - （同InstructNav）实例模块（检测、分割、追踪、是否动态物）的封装和定制化

## Sim2Real
  - rgbd对齐误差
  - lidar噪声
  - 定位噪声
  - segmentor噪声
  - 动态物处理

## System impl
  - grid_map migrate
  - To planner
  - To controller
  - nav2行为树impl
  - 预探索地图和未探索法的结合
  - How to 建模世界，what level, which hierarchicals, how align。(sparse, graph, grid, feature aggregation)

