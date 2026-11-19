# DNSMOS P.835 独立打分说明

交付物：

- `score_dnsmos_p835.py`
- `score_dnsmos_p835.yaml`（干净独立配置，无流水线字段）
- `sig_bak_ovr.onnx`

## 环境

```bash
pip install onnxruntime numpy soundfile pyyaml
# 音频不是 16 kHz 时再装：
pip install librosa
```

## 权重

官方 DNSMOS P.835 primary ONNX，文件名：`sig_bak_ovr.onnx`。

默认写在 `score_dnsmos_p835.yaml` 的 `model_path`（与服务器现用路径一致）：

```text
/data2/data-cp/lizi/DNS-Challenge-master/DNSMOS/DNSMOS/sig_bak_ovr.onnx
```

改成本机可读路径，或运行时用 `--model` 覆盖。

## 三个分数

对单段语音给出 P.835 感知质量分（约 1–5，越高越好）：

| 字段 | 含义 |
|------|------|
| `sig` | SIG：语音本身清晰度 / 失真程度 |
| `bak` | BAK：背景噪声干扰程度（越高表示背景越干净） |
| `ovrl` | OVRL：整体听感质量 |

## 用法

```bash
python scripts/score_dnsmos_p835.py \
  --config scripts/score_dnsmos_p835.yaml \
  /path/to/audio.wav

# 或覆盖权重路径
python scripts/score_dnsmos_p835.py \
  --config scripts/score_dnsmos_p835.yaml \
  --model /path/to/sig_bak_ovr.onnx \
  /path/to/audio.wav
```
