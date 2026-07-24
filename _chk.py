import ast, sys
p = "/Users/chunter/code/curtis/Synthetic2/infra/cloud_providers/_ssh.py"
try:
    ast.parse(open(p).read())
    print("SSH_PY_SYNTAX_OK")
except SyntaxError as e:
    print("SYNTAX_ERROR", e.lineno, e.msg)
    sys.exit(2)
