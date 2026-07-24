import re, io, sys, hashlib
P = "/Users/chunter/code/curtis/Synthetic2/infra/cloud_providers/_ssh.py"
src = open(P).read()

new_block = open("/tmp/_new_resolver.txt").read().rstrip("\n") + "\n"

# Anchors: from the line `PYBIN=""` up to (but NOT including) the
# `echo "[wrapper] using PYBIN=...` line that follows the resolver.
start_anchor = 'PYBIN=""\n'
end_anchor = 'echo "[wrapper] using PYBIN='

i = src.find(start_anchor)
j = src.find(end_anchor)
if i == -1 or j == -1 or j < i:
    print("ANCHOR_FAIL", i, j); sys.exit(2)

before = src[:i]
after = src[j:]
patched = before + new_block + "\n" + after

open("/tmp/_ssh_patched.py", "w").write(patched)
print("PATCHED bytes_old=%d bytes_new=%d" % (len(src), len(patched)))
print("OUT=/tmp/_ssh_patched.py")
