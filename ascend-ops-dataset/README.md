# AscendC task corpus

本目录包含 682 条 AscendC 算子任务语料，以及 5 个具备配套工程和测试脚本的代表算子子集。

## Counts

- `final/test.json`：682 条任务记录。
- `has_test_scripts=true`：5 条，分别为 Add、Relu、Sigmoid、Fmod、Asinh。
- `has_test_scripts=false`：677 条，仅可用于任务语料研究，不能直接声明完成真机精度验证。

## Files

- `final/test.json`：统一任务记录。
- `final/operator_list.json`：算子索引与上游链接。
- `final/test_selected.json`：精选任务子集。
- `sources/desktop-5-ops/`：5 个代表算子的参考工程。
- `build_dataset.py`：数据整理脚本。
- `select_subset.py`：子集筛选脚本。

单条记录以实际 JSON 字段为准，主要包括：

```json
{
  "name": "ascend_add",
  "category": "...",
  "difficulty": "...",
  "description": "...",
  "prompt": "...",
  "reference_kernel": "...",
  "reference_host": "...",
  "test_gen_script": "...",
  "test_verify_script": "...",
  "original_source": "desktop-5-ops",
  "has_test_scripts": true
}
```

运行 `python run.py dataset-info` 或测试 `tests/test_dataset.py` 可复核数量边界。

## License boundary

数据中的上游参考实现、CANN 工程与文档片段版权归原权利人所有，不适用根目录 MIT License。详见 [LICENSE_DATA.md](LICENSE_DATA.md) 和根目录 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。公开发布前应再次核对上游许可和署名要求。
