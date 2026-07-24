import sys
sys.path.insert(0, "/Users/chunter/code/curtis/Synthetic2/infra")
from cloud_providers import _ssh
import inspect
sig = inspect.signature(_ssh.build_remote_wrapper)
print("PARAMS:", list(sig.parameters))
