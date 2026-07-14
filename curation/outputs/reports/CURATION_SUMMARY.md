# CURATION_SUMMARY(首次归档交付, 2026-07-14)

## 交付性质与豁免声明(重要)
本批为**机器筛选交付**, 经批准豁免"每样本≥2人独立标注"要求(2026-07-14, 用于先行实验)。
所有行的 `annotation_count=0`、`annotator_agreement=0.0` 均为**如实记录**;
`use_in_main` 按下述豁免标准计算。**双人标注版将在后续批次补齐。**

## 总量
- unified: 4754 | Conflict: 585(use_in_main=460) | Aligned: 4169(use_in_main=4089) | Ambiguous: 0

## 数据源与筛选漏斗
### 1. CH-SIMS v2(s)(真实数据, source_is_generated=false)
- 原始 4403 条(视频有人工 V/T/A/联合四标签)
- 初筛(规则, 阈值|v|>=0.4, 冲突差>=0.8): VT 冲突175/一致1888, VA 冲突63/一致2141
- 冲突候选 238 -> 综艺花字检测剔除 42(清单见 variety_discarded.json) -> 196
- 196 条冲突视频已**裁除底部字幕**(裁底部18%, 防文本泄漏进视觉通道), media_paths 指向服务器裁剪副本
- LLM 精筛(Gemini 3.1, OpenRouter): 全部冲突候选 + 部分 VT 一致样本; VA 一致样本因 API 额度终止未覆盖
- 豁免标准: Conflict use_in_main = Gemini确认为Conflict且无质量旗标; Aligned use_in_main = 无真实LLM反对意见
- 主要剔除原因: 初筛不清晰/模糊(约2340+2199条), 综艺花字42条, Gemini判非冲突

### 2. 生成数据(LTX-2.3渲染, source_is_generated=true, 归档池=单人质量审核通过)
- accept_a_svt 337 条Conflict(VT) / accept_a_va 52 条Conflict(VA) / accept_c_svt 140 条Aligned
- 标签依据: 设计标签(gt/surface情绪按构造已知) + DeepSeek判定台词极性以分派通道 + 人工质量审核(可用/淘汰)
- 通道极性冲突塌缩的 0 条降级为 Ambiguous(use_in_main=false)
- 生成来源信息保留在每行 generation_info(archetype/seed/gt/surface/rationale)
- 质量漏斗(历史): a_svt 渲染约1900+, 单人审核淘汰率约75%; dialogue无明确情绪自动预筛(DeepSeek)累计剔除约880条

## 标注员
- 本批人工标注数=0(豁免); human_annotations.jsonl 仅含流程冒烟测试记录
- 8888 质量审核(生成数据 accept/reject)由团队成员执行(单人)

## 媒体路径
- 全部为服务器 172.17.43.38 上的绝对路径, 可直接访问
- CH-SIMS 冲突+一致样本: /home/team/lvshuyang/TAFFC/mprisk/curation/outputs/media_cropped/
  (全部 2537 个源视频已裁除底部字幕18%, 防文本泄漏进视觉通道)
- 一致样本中 363 条带 quality_flags=['variety_source_suspect'](与已确认花字视频同
  video_id 家族, 未逐条视觉复检, 实验时可按需过滤)
- 生成数据: /home/team/lvshuyang/prompt-make/output/accept_*/
