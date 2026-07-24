import sys
sys.path.insert(0, "/Users/chunter/code/curtis/Synthetic2/infra")
from cloud_providers import _ssh
fin = {"b2_key_id":"KID","b2_key":"KEY","b2_bucket":"lab-sm",
       "runpod_api_key":"RPA","pod_id":"POD123","run_id":"run-xyz","hf_token":"hf_x"}
try:
    w = _ssh.build_remote_wrapper(benchmark="trace", smoke=True, extra_args="", remote_finalize=fin)
except TypeError as e:
    # signature may take remote_finalize differently; try minimal
    w = _ssh.build_remote_wrapper("trace", True, "", fin)
open("/Users/chunter/code/curtis/Synthetic2/_wrapper_rendered.sh","w").write(w)
print("RENDERED_OK bytes", len(w))
