#!/usr/bin/env python3
"""Pull the (already NVVM-lowered) gpu.module out of a gpu.container_module so mlir-translate
can emit plain LLVM IR for the kernels: the host side of the module is dropped."""
import re, sys
src = open(sys.argv[1]).read()
m = re.search(r'gpu\.module @\w+\s*(\[[^\]]*\])?\s*\{', src)
if not m: sys.exit("no gpu.module found")
i = m.end(); depth = 1; j = i
while depth:
    c = src[j]; depth += (c == '{') - (c == '}'); j += 1
body = src[i:j-1]
out = "module {\n" + body + "\n}\n"
open(sys.argv[2], 'w').write(out)
