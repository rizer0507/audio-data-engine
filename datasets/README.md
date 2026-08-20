# 存放数据集 Manifest

Manifest 是数据集的唯一真相源，支持 JSONL 和 Parquet 两种格式。

## 命名约定

```
raw_YYYYMMDD.parquet      # 原始导入
processed_YYYYMMDD.parquet # 处理后的版本
```

## 使用

```bash
# 从原始目录导入
audio-data ingest /path/to/wavs --name raw_20260820

# 查看统计
audio-data stats raw_20260820
```
