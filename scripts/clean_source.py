import re
import sys

if len(sys.argv) != 3:
    print("Usage: python scripts/clean_source.py input.c output.c")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

with open(input_file, "r") as f:
    code = f.read()

# Compiler/project-specific attributes
attributes = [
    "av_always_inline",
    "av_noinline",
    "av_cold",
    "coroutine_fn",
    "always_inline",
    "av_unused",
    "av_noreturn",
    "QEMU_NORETURN",
    "QEMU_WARN_UNUSED_RESULT",
    "G_GNUC_UNUSED",
    "attribute_align_arg",
    "STATUS_PARAM",
    "av_restrict",
    "av_flatten",
    "CUDAAPI",
    "WINAPI",
    "CALLBACK",
    "OPPROTO",
]

for attr in attributes:
    code = re.sub(
        r"\b" + re.escape(attr) + r"\b\s*",
        "",
        code
    )

# Macros that wrap the function name.
# FUNCC(foo) -> foo
# FUNC(foo)  -> foo
# HELPER(foo) -> foo
# RENAME(foo) -> foo
for macro in ["FUNCC", "FUNC", "HELPER", "RENAME"]:
    code = re.sub(
        r"\b" + macro +
        r"\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)",
        r"\1",
        code
    )

# Remove function attributes that take arguments:
# GCC_FMT_ATTR(2, 3)
# etc.
code = re.sub(
    r"\bGCC_FMT_ATTR\s*\([^)]*\)\s*",
    "",
    code
)

# Remove similar QEMU attribute macros with arguments.
code = re.sub(
    r"\bQEMU_[A-Z_]+\s*\([^)]*\)\s*",
    "",
    code
)

# Remove simple QEMU macro attributes.
code = re.sub(
    r"\bQEMU_[A-Z_]+\b\s*",
    "",
    code
)

with open(output_file, "w") as f:
    f.write(code)

print(f"Cleaned: {input_file} -> {output_file}")
