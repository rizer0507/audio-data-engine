# Sophon-Realtime-Tag 内网构建与部署

在现有 A800 服务器上，用内网 PyPI 镜像直接创建 Conda 环境并安装依赖，不走外网打包迁移。

硬件评估、目标版本、GPU 选择、启动参数与验收标准，与 `Daily/2026-11-10.md` 外网方案一致；差异仅在「依赖安装方式」和「权重路径」。

内网 PyPI：

```
http://maven.paic.com.cn/repository/pypi/simple
--trusted-host maven.paic.com.cn
```

## 一、现有服务器环境评估

| 项目                   | 当前环境             | 评估              |
| ---------------------- | -------------------- | ----------------- |
| 操作系统               | Ubuntu 22.04.5 LTS   | 满足              |
| 架构                   | x86_64，glibc 2.35   | 满足              |
| GPU                    | 8 × NVIDIA A800 80GB | 基本满足          |
| GPU Compute Capability | 8.0                  | 满足              |
| NVIDIA Driver          | 595.71.05            | 支持 CUDA 13.0    |
| 驱动显示 CUDA          | 13.2                 | 满足              |
| 系统内存               | 2TB                  | 充足              |
| Python                 | 3.11.9               | 建议新建 3.12     |
| PyTorch                | 2.10.0+cu128         | 新环境独立安装    |
| vLLM                   | 0.19.0               | 新环境使用 0.30.0 |
| 根分区可用空间         | 82GB                 | 不建议用于新环境  |

### GPU 选择

部署前重新检查显存占用。首选 GPU 6，端口 8000。官方 Tag 单实例约 65GB，A800 80GB 基本满足，最终以启动测试为准。

```
nvidia-smi -i 6
```

## 二、目标软件版本

| 组件              | 目标版本             | 来源                 |
| ----------------- | -------------------- | -------------------- |
| Python            | 3.12.x               | 新 Conda 环境        |
| vLLM              | 0.30.0               | 内网 PyPI            |
| vLLM-Omni         | 0.30.0               | 内网 PyPI            |
| duplexjev-vllm    | 0.3.0                | 内网 PyPI            |
| Sophon Tag Plugin | 0.3.0                | 本地 `tag/plugin`    |
| PyTorch           | 由 vLLM 依赖解析确定 | 内网 PyPI CUDA 13.0  |
| Triton            | 由 vLLM 依赖解析确定 | 内网 PyPI            |
| TensorRT          | 首次部署不安装       | 后续可选             |

不要用仓库根目录的 `requirements-engine.txt` 覆盖这套依赖。Tag 部署以 `tag/install.sh` 为基准。

## 三、最终目录规划

```
/data2/data-cp/lizi/
│
├── conda_envs/
│   ├── asr/                   # 原有环境，不修改
│   └── sophon/                # 新增 Sophon 环境
│
├── sophon-realtime/
│   └── sophon-realtime-main/  # 现场解压后的官方源码
│       └── tag/               # Tag 部署目录（不是仓库根目录）
│           ├── plugin/
│           ├── config/
│           ├── start.sh
│           ├── stop.sh
│           └── client_example.py
│
└── sophon_runtime/
    ├── cache/                 # 编译及模型缓存
    └── output/                # 测试输出

/data2/data-cp/models/
└── Sophon-Realtime-tag/       # 模型权重（固定路径）
```

先确认数据盘空间：

```
df -h /data2
df -i /data2

du -sh /data2/data-cp/lizi/conda_envs/asr
```

建议在数据盘为新环境、编译缓存预留至少 40GB。

## 四、内网创建 Conda 环境并安装依赖

所有操作在 A800 内网服务器执行。不破坏已有 ASR / Kimi / GLM / Qwen 环境。

### 4.1 建立目录和变量

```
export BASE=/data2/data-cp/lizi

export ENV_DIR=$BASE/conda_envs/sophon
export RUNTIME_DIR=$BASE/sophon_runtime

# 现场源码在 sophon-realtime-main 下；Tag 插件在 tag/ 子目录
export TAG_DIR=$BASE/sophon-realtime/sophon-realtime-main/tag
export MODEL_DIR=/data2/data-cp/models/Sophon-Realtime-tag

export PIP_INDEX_URL=http://maven.paic.com.cn/repository/pypi/simple
export PIP_TRUSTED_HOST=maven.paic.com.cn
export PYTHONNOUSERSITE=1
```

检查：

```
df -h /data2

test -f "$TAG_DIR/start.sh" && echo "源码存在"
test -f "$MODEL_DIR/config.json" && echo "模型配置存在"
```

任一检查失败先改路径，不要继续。

建立目录：

```
mkdir -p \
    "$ENV_DIR" \
    "$RUNTIME_DIR/cache" \
    "$RUNTIME_DIR/output"
```

### 4.2 创建最小 Python 3.12 环境

重要：`PIP_INDEX_URL` / `PIP_TRUSTED_HOST` 只作用于 pip，不作用于 conda。

`conda create` 会打外网 `repo.anaconda.com`；`rsync` 整环境会把 ASR 里的 torch/vllm 等全部拷过来（数 GB、上万文件），Sophons 不需要。

正确做法：借用已有 3.12 的解释器，`python -m venv` 建干净环境（只有标准库 + pip），再靠内网 pip 装 Sophon 依赖。

若刚才已经在 rsync，先停掉并清掉半成品：

```
# 终端里 Ctrl+C 停掉 rsync
rm -rf "$ENV_DIR"
```

然后：

```
export SRC_ENV=/data2/data-cp/zk/env_hub/miniconda312_qw3tts
# 若 SRC 不可用，可改用：
# export SRC_ENV=/data2/data-cp/lizi/env_hub/miniconda312_qw3tts_vllm020

"$SRC_ENV/bin/python" --version   # 确认是 3.12.x

mkdir -p "$(dirname "$ENV_DIR")"

# 只拿解释器，不拷贝 site-packages
"$SRC_ENV/bin/python" -m venv "$ENV_DIR"

"$ENV_DIR/bin/python" -c "import sys; print(sys.executable); print(sys.prefix)"
"$ENV_DIR/bin/python" --version
"$ENV_DIR/bin/python" -m pip --version
```

期望：

```
/data2/data-cp/lizi/conda_envs/sophon/bin/python
```

激活与启动脚本兼容（venv 没有真正的 conda prefix，手动对齐即可）：

```
source "$ENV_DIR/bin/activate"
export CONDA_PREFIX="$ENV_DIR"
which python
```

更新安装工具（一律走内网 pip 镜像）：

```
"$ENV_DIR/bin/python" -m pip install --upgrade \
    pip setuptools wheel \
    --index-url "$PIP_INDEX_URL" \
    --trusted-host "$PIP_TRUSTED_HOST"
```

一律用 `"$ENV_DIR/bin/python" -m pip`，不要直接 `pip`、不要 `--user`。  
若 `python -m venv` 报缺少 ensurepip，再说；多数现有 3.12 conda 解释器可直接建 venv。

备选（不推荐默认）：只有在你明确要「整环境克隆升级」时才 `rsync -a`，那是总手册里 vLLM 升版场景，不是 Sophon 最小底座。

### 4.3 安装完整推理依赖

```
"$ENV_DIR/bin/python" -m pip install \
    "vllm==0.30.0" \
    "vllm-omni==0.30.0" \
    "duplexjev-vllm==0.3.0" \
    --index-url "$PIP_INDEX_URL" \
    --trusted-host "$PIP_TRUSTED_HOST"
```

不手动固定 PyTorch、Triton 版本，由 vLLM 依赖自动解析。vLLM 0.30.0 默认 CUDA 13.0，与当前 595 驱动匹配。

音频与离线 VAD 辅助依赖：

```
"$ENV_DIR/bin/python" -m pip install \
    onnxruntime \
    soundfile \
    librosa \
    --index-url "$PIP_INDEX_URL" \
    --trusted-host "$PIP_TRUSTED_HOST"
```

首次部署不安装 TensorRT。

若内网镜像缺包，先在镜像仓库确认是否有对应 wheel；有本地 wheel 时改用：

```
"$ENV_DIR/bin/python" -m pip install /path/to/xxx.whl
```

不要改走公网 PyPI。

### 4.4 验证环境导入

```
"$ENV_DIR/bin/python" - <<'PY'
import sys
import torch
import vllm
import vllm_omni
import duplexjev_vllm
import onnxruntime
import soundfile

print("Python:", sys.version)
print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("vLLM:", vllm.__version__)
print("vLLM-Omni:", vllm_omni.__version__)
print("ONNXRuntime:", onnxruntime.__version__)
print("依赖模块导入成功")
PY

"$ENV_DIR/bin/python" -m pip check
```

`pip check` 不能有依赖冲突。CUDA 推理验证放到启动阶段。

可选落盘版本清单：

```
"$ENV_DIR/bin/python" -m pip freeze \
    > "$RUNTIME_DIR/requirements-lock.txt"
```

### 4.5 安装 Sophon 官方插件

```
cd "$TAG_DIR"

"$ENV_DIR/bin/python" -m pip install \
    --no-index \
    --no-deps \
    --no-build-isolation \
    -e ./plugin
```

插件必须从 `tag/plugin` 安装，不是仓库根目录其他包。

验证：

```
conda activate "$ENV_DIR"

python - <<'PY'
import torch
import vllm
import vllm_omni
import duplexjev_vllm
import sophon_realtime

print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("vLLM:", vllm.__version__)
print("vLLM-Omni:", vllm_omni.__version__)
print("Sophon Plugin:", sophon_realtime.__version__)
print("CUDA Available:", torch.cuda.is_available())

assert torch.cuda.is_available(), "CUDA 不可用"
print("基础环境检查通过")
PY

python -m pip check
```

## 五、启动前配置

### 5.1 检查模型权重是否完整

权重固定目录：`/data2/data-cp/models/Sophon-Realtime-tag`

```
du -sh "$MODEL_DIR"

ls -lh "$MODEL_DIR"/config.json

ls -lh \
    "$MODEL_DIR/talker/flow.decoder.estimator.fp32.onnx"

ls -lh \
    "$TAG_DIR/config/silero_vad.onnx"

find -L "$MODEL_DIR" -type l -print
```

`find` 若输出断开软链接必须修复；必需文件缺失需补全。

### 5.2 配置所有缓存走数据盘

根分区紧张时必须做：

```
export HF_HOME=$RUNTIME_DIR/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export XDG_CACHE_HOME=$RUNTIME_DIR/cache/xdg
export TORCH_HOME=$RUNTIME_DIR/cache/torch
export TORCH_EXTENSIONS_DIR=$RUNTIME_DIR/cache/torch_extensions
export TRITON_CACHE_DIR=$RUNTIME_DIR/cache/triton
export VLLM_CACHE_ROOT=$RUNTIME_DIR/cache/vllm

export TMPDIR=$RUNTIME_DIR/cache/tmp

mkdir -p \
    "$HF_HOME" \
    "$XDG_CACHE_HOME" \
    "$TORCH_HOME" \
    "$TORCH_EXTENSIONS_DIR" \
    "$TRITON_CACHE_DIR" \
    "$VLLM_CACHE_ROOT" \
    "$TMPDIR"
```

### 5.3 预置 CosyVoice3 的离线语音 tokenizer

2026-11-10 的启动日志中，6179 实例的 Stage 0（思考器）已加载权重、建立 KV cache 并完成预热。Stage 1（CosyVoice3 发声模块）在构造多模态预热输入时调用 s3tokenizer 的 `load_model("speech_tokenizer_v3_25hz")`。缓存未命中时，s3tokenizer 用 `urllib` 访问魔搭，内网 DNS 解析失败，随后 API 进程报引擎初始化失败。`HF_HUB_OFFLINE` 和 `TRANSFORMERS_OFFLINE` 只影响相应库的加载流程，s3tokenizer 会自行处理下载。

Tag 权重包的 `talker/speech_tokenizer_v3.onnx` 已包含所需文件。本地副本的 SHA-256 已与 s3tokenizer 0.3.0 的预期值核对。将它以加载器要求的名字链接到 `$XDG_CACHE_HOME/s3tokenizer`，无需另行下载。以下使用 4.1 中的路径，且必须在之后启动服务的同一 shell 中设置 `XDG_CACHE_HOME`。

```bash
export XDG_CACHE_HOME="$RUNTIME_DIR/cache/xdg"

(
    set -e
    source_onnx="$MODEL_DIR/talker/speech_tokenizer_v3.onnx"
    cache_dir="$XDG_CACHE_HOME/s3tokenizer"
    expected_sha256=23236a74175dbdda47afc66dbadd5bcb41303c467a57c261cb8539ad9db9208d

    printf '%s  %s\n' "$expected_sha256" "$source_onnx" | sha256sum --check -
    mkdir -p "$cache_dir"
    ln -sfnT -- "$source_onnx" "$cache_dir/speech_tokenizer_v3_25hz.onnx"
)
```

校验成功后再启动。文件缺失或哈希不符时，从现有 Tag 权重副本补齐 `talker/speech_tokenizer_v3.onnx`，再重跑本节命令。缓存链接指向模型文件，服务进程需要能读取该文件。当前已核对文件和加载器逻辑，内网服务器重启和音频验收待执行。 [vLLM-Omni 0.30.0 调用源码](https://github.com/vllm-project/vllm-omni/blob/v0.30.0/vllm_omni/model_executor/models/cosyvoice3/cosyvoice3.py#L243-L262) · [s3tokenizer 缓存源码](https://github.com/xingchensong/S3Tokenizer/blob/main/s3tokenizer/__init__.py)

### 5.4 准备支持 Conda 的启动脚本

官方 `start.sh` 固定使用 `tag/.venv/bin`。复制后改 Conda：

```
cd "$TAG_DIR"

cp start.sh start_conda.sh

python - <<'PY'
from pathlib import Path

p = Path("start_conda.sh")
s = p.read_text()

old = "PY=$HERE/.venv/bin"
new = "PY=${CONDA_PREFIX}/bin"

assert old in s, "没有找到官方脚本中的 Python 路径，请核对版本"

p.write_text(s.replace(old, new, 1))
print("start_conda.sh 修改成功")
PY

chmod +x start_conda.sh
```

只改 Python 路径，其余保持官方行为。

## 六、使用 GPU 6 启动

```
nvidia-smi -i 6

cd "$TAG_DIR"

source "$ENV_DIR/bin/activate"
export CONDA_PREFIX="$ENV_DIR"

MODEL_DIR="$MODEL_DIR" \
GPU=6 \
PORT=8000 \
TRT=0 \
MPS=0 \
bash start_conda.sh
```

- `TRT=0`：首次不启用 TensorRT
- `MPS=0`：首次不启用 CUDA MPS
- `GPU=6` 时 `CUDA_VISIBLE_DEVICES=6` 会把物理卡映射为进程内 device 0，YAML 里三个 `devices: "0"` 不用改成 6

### 6.1 查看启动日志

```
tail -f "$TAG_DIR/logs/server_8000.log"
```

### 6.2 验证 HTTP 服务

```
curl -v http://127.0.0.1:8000/health

curl -s http://127.0.0.1:8000/v1/models
```

### 6.3 音频测试

```
cd "$TAG_DIR"

python client_example.py \
    --url http://127.0.0.1:8000 \
    --wav examples/hello_zh.wav \
    --lang zh \
    --voice vivian_zh \
    --out "$RUNTIME_DIR/output/reply.wav"
```

### 6.4 停止服务

```
cd "$TAG_DIR"

PORT=8000 bash stop.sh
```

只停 8000 端口对应实例，不影响其他 GPU 上的服务。

## 七、部署成功后再考虑的优化

| 优化项    | 初次启动      | 稳定运行后                 |
| --------- | ------------- | -------------------------- |
| TensorRT  | 关闭          | 可开启，改善 Flow 解码效率 |
| CUDA MPS  | 关闭          | 可开启，改善多阶段并发     |
| 推理并发  | 先测试单请求  | 逐步提高                   |
| GPU       | GPU 6 单卡    | GPU 7 可运行另一实例       |
| vLLM 配置 | 官方默认 YAML | 根据显存和延迟调整         |

启用 TensorRT 时再装：

```
tensorrt-cu13==10.13.3.9
tensorrt-cu13-bindings==10.13.3.9
tensorrt-cu13-libs==10.13.3.9
```

同样走内网镜像；首次跑通前不必安装。

## 八、部署验收标准

- [ ] Python 3.12 环境能正常激活（`$ENV_DIR`）
- [ ] vLLM 0.30.0 与 vLLM-Omni 0.30.0 导入成功
- [ ] Sophon 专用插件注册成功
- [ ] PyTorch 能识别 A800 GPU
- [ ] 模型与 ONNX 文件完整（`/data2/data-cp/models/Sophon-Realtime-tag`）
- [ ] S3Tokenizer ONNX 文件校验成功，缓存链接与启动进程的 `XDG_CACHE_HOME` 一致
- [ ] GPU 6 上 `/health` 返回成功
- [ ] 中文音频请求生成回复 WAV

## 九、与外网方案差异摘要

| 项       | 外网方案                         | 本内网方案                                      |
| -------- | -------------------------------- | ----------------------------------------------- |
| 依赖获取 | 外网装好后 `conda-pack` 拷入     | 已有 3.12 解释器 `venv` + 内网 `pip install`    |
| PyPI     | 公网                             | `maven.paic.com.cn`（只影响 pip，不影响 conda） |
| 建环境   | `conda create` / conda-pack      | 禁止 `conda create`；也不要整目录 rsync ASR 环境 |
| 权重路径 | `$BASE/models/Sophon-Realtime-32B-Tag` | `/data2/data-cp/models/Sophon-Realtime-tag` |
| Conda 环境 | `/data2/data-cp/lizi/conda_envs/sophon` Python 3.12 | 相同 |
| GPU/端口 | GPU 6 / 8000                     | 相同                                            |
| TRT/MPS  | 首次关闭                         | 相同                                            |
