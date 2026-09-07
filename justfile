# CS336 assignment1-basics
#
# 常用：
#   just            列出所有 recipe
#   just check      写完代码先跑这个（格式 + lint + 类型检查，与 lefthook 一致）
#   just fix        自动修掉能修的（ruff --fix + format）
#   just test       跑全部测试

# ty 的排除项与 lefthook.yml 保持一致
ty_flags := "--exclude tests/conftest.py --exclude cs336_basics/pretokenization_example.py"

default:
    @just --list

# 手动自检：三项全跑，中间失败也继续，最后汇总
check:
    #!/usr/bin/env bash
    set -uo pipefail
    fail=0
    echo "── ruff format --check ──────────────────────────────"
    uv run ruff format --check . || fail=1
    echo "── ruff check ──────────────────────────────────────"
    uv run ruff check . || fail=1
    echo "── ty check ────────────────────────────────────────"
    uv run ty check {{ ty_flags }} || fail=1
    echo "────────────────────────────────────────────────────"
    if [ "$fail" -ne 0 ]; then
        echo "✗ check 未通过（格式问题可用 just fix 修）"
        exit 1
    fi
    echo "✓ check 通过"

# 只跑类型检查
typecheck:
    uv run ty check {{ ty_flags }}

# 只跑 lint
lint:
    uv run ruff check .

# 只检查格式，不改文件
fmt-check:
    uv run ruff format --check .

# 自动修复：ruff --fix + 格式化
fix:
    uv run ruff check --fix .
    uv run ruff format .

# 跑测试，可透传参数：just test tests/test_train_bpe.py -x
test *args:
    uv run pytest {{ args }}

# 当前在做的 BPE 训练测试
test-bpe *args:
    uv run pytest tests/test_train_bpe.py {{ args }}

# 提交前的完整检查：静态检查 + 测试
ci: check test
