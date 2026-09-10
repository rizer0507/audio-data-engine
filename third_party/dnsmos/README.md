# DNSMOS P.835 vendor assets

Place the official Microsoft DNSMOS P.835 **primary** ONNX here:

```text
third_party/dnsmos/sig_bak_ovr.onnx
```

Then:

```bash
pip install onnxruntime
# or
pip install .[dnsmos]
```

`quality.dnsmos` fails fast at startup if the model file is missing.
Do not commit large weight files unless the team explicitly vendors them.
Thresholds live in `configs/quality/dnsmos_p835.yaml` (`calibrated: false` until human calibration passes).
