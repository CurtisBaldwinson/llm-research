import sys, tempfile, subprocess, os
sys.path.insert(0, "/Users/chunter/code/curtis/Synthetic2/infra")
from cloud_providers import _ssh
fin = {"b2_key_id":"KID","b2_key":"KEY","b2_bucket":"lab-sm",
       "runpod_api_key":"RPA","pod_id":"POD123","run_id":"run-xyz","hf_token":"hf_x"}
try:
    w = _ssh.build_remote_wrapper(benchmark="trace", smoke=True, extra_args="", remote_finalize=fin)
except TypeError:
    w = _ssh.build_remote_wrapper("trace", True, "", fin)
# write to /tmp (allowed), then bash -n
tf = tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False, dir="/tmp")
tf.write(w); tf.close()
r = subprocess.run(["bash","-n",tf.name], capture_output=True, text=True)
print("BASH_LINT_RC", r.returncode)
print(r.stderr[-500:] if r.stderr else "no stderr")
# show the PYBIN block + finalize python line as rendered
for ln in w.splitlines():
    if "PYBIN" in ln or "pip install" in ln or "pod_finalize.py" in ln or "CUDA OK" in ln:
        print(">>", ln)
os.unlink(tf.name)
