P = "/Users/chunter/code/curtis/Synthetic2/infra/cloud_providers/_ssh.py"
src = open(P).read()
orig = src

# --- 1) revert install_cmd: "$PYBIN" -m pip  ->  pip ---
src = src.replace(
    '        "    | xargs -r \\"$PYBIN\\" -m pip install --quiet --no-cache-dir; "',
    '        "    | xargs -r pip install --quiet --no-cache-dir; "'
)

# --- 2) revert entry exec: "$PYBIN" -u  ->  python3 -u ---
src = src.replace(
    'stdbuf -oL -eL "$PYBIN" -u "$ENTRY" {smoke_flag} {extra_args}',
    'stdbuf -oL -eL python3 -u "$ENTRY" {smoke_flag} {extra_args}'
)

# --- 3) revert finalize call: "$PYBIN"  ->  python3 ---
src = src.replace(
    '        "$PYBIN" "{REMOTE_FINALIZE_DIR}/pod_finalize.py" \\',
    '        python3 "{REMOTE_FINALIZE_DIR}/pod_finalize.py" \\'
)

# --- 4) remove the whole resolver block, restore v1's single CUDA-check line ---
# The block starts at the comment marker and ends right before the CUDA check
# line that uses "$PYBIN". We replace from the marker through the
# `echo "[wrapper] using PYBIN=..."` blank line, and swap the "$PYBIN" CUDA
# check back to python3.
start = src.find('# --- resolve a Python interpreter that actually has torch')
# the CUDA check line (now using "$PYBIN") is what v1 had as python3:
cuda_pybin = '"$PYBIN" -c "import torch; assert torch.cuda.is_available(), \'CUDA not detected\'; print(\'CUDA OK:\', torch.cuda.get_device_name(0))" || {{'
cuda_idx = src.find(cuda_pybin)
if start == -1 or cuda_idx == -1:
    print("MARKER_FAIL start=%d cuda=%d" % (start, cuda_idx)); raise SystemExit(2)

before = src[:start]
after = src[cuda_idx + len(cuda_pybin):]  # everything after the CUDA-check line opener
v1_cuda = 'python3 -c "import torch; assert torch.cuda.is_available(), \'CUDA not detected\'; print(\'CUDA OK:\', torch.cuda.get_device_name(0))" || {{'
src = before + v1_cuda + after

open("/tmp/_ssh_reverted.py","w").write(src)
print("REVERTED old=%d new=%d delta=%d" % (len(orig), len(src), len(src)-len(orig)))
