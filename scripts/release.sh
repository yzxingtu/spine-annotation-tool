#!/usr/bin/env bash
# ==============================================================================
# spine-annotation-tool 发布脚本
#
# 工作流程:
#   1. 检查 git 工作区是否干净（无未提交改动）
#   2. 检查并完成本地未推送的提交
#   3. 修改全局版本号 (src/spine_annotator/__init__.py + pyproject.toml)
#   4. 提交版本号改动
#   5. 推送 main 分支
#   6. 打 tag 并推送，触发 GitHub Actions 构建
#
# 用法:
#   ./scripts/release.sh <version>
#   例: ./scripts/release.sh 0.2.0
# ==============================================================================
set -euo pipefail

# ---------- 颜色 ----------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()      { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()     { echo -e "${RED}[ERROR]${NC} $*" >&2; }
fatal()   { err "$*"; exit 1; }

# ---------- 进入项目根目录 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYPROJECT_FILE="pyproject.toml"
INIT_FILE="src/spine_annotator/__init__.py"

# ---------- 校验参数 ----------
if [[ $# -lt 1 ]]; then
  err "用法: $0 <version>"
  err "示例: $0 0.2.0"
  exit 1
fi

NEW_VERSION="$1"
TAG="v${NEW_VERSION}"

# 简单的语义化版本校验
if ! [[ "${NEW_VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][a-zA-Z0-9]+)?$ ]]; then
  fatal "版本号格式不合法: ${NEW_VERSION} (期望格式如 0.2.0 / 1.0.0-rc1)"
fi

info "准备发布版本: ${NEW_VERSION}  (tag: ${TAG})"

# ---------- 1. 检查 git 工作区是否干净 ----------
info "[1/6] 检查 git 工作区..."
if [[ -n "$(git status --porcelain)" ]]; then
  err "工作区有未提交改动，请先 commit 或 stash:"
  git status --short
  exit 1
fi
ok "工作区干净"

# ---------- 2. 检查/完成推送 ----------
info "[2/6] 检查待推送提交..."
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ -z "${CURRENT_BRANCH}" || "${CURRENT_BRANCH}" == "HEAD" ]]; then
  fatal "当前不在任何分支上 (HEAD detached)"
fi
info "当前分支: ${CURRENT_BRANCH}"

# 拉取远程最新
git fetch origin --tags --prune

# 校验 tag 是否已存在
if git rev-parse -q --verify "refs/tags/${TAG}" >/dev/null; then
  fatal "本地已存在 tag ${TAG}，请先删除或换版本号"
fi
if git ls-remote --tags origin "refs/tags/${TAG}" | grep -q .; then
  fatal "远端已存在 tag ${TAG}，请换版本号"
fi

# 检查是否有 upstream
if ! git rev-parse --abbrev-ref --symbolic-full-name @{u} >/dev/null 2>&1; then
  warn "当前分支无 upstream，将设置为 origin/${CURRENT_BRANCH}"
  git push --set-upstream origin "${CURRENT_BRANCH}"
else
  AHEAD=$(git rev-list --count "@{u}..HEAD" || echo 0)
  BEHIND=$(git rev-list --count "HEAD..@{u}" || echo 0)
  if [[ "${BEHIND}" -gt 0 ]]; then
    fatal "本地落后远端 ${BEHIND} 个提交，请先 pull/rebase"
  fi
  if [[ "${AHEAD}" -gt 0 ]]; then
    info "本地领先远端 ${AHEAD} 个提交，开始推送..."
    git push origin "${CURRENT_BRANCH}"
    ok "已推送 ${AHEAD} 个待推送提交"
  else
    ok "本地已与远端同步"
  fi
fi

# ---------- 3. 修改版本号 ----------
info "[3/6] 修改版本号 -> ${NEW_VERSION}"

# 跨平台 sed -i (macOS 与 GNU 不同)
sed_inplace() {
  if [[ "$(uname)" == "Darwin" ]]; then
    sed -i '' "$@"
  else
    sed -i "$@"
  fi
}

# pyproject.toml: version = "x.y.z"
if ! grep -qE '^version[[:space:]]*=' "${PYPROJECT_FILE}"; then
  fatal "${PYPROJECT_FILE} 中未找到 version 字段"
fi
sed_inplace -E "s|^version[[:space:]]*=[[:space:]]*\".*\"|version = \"${NEW_VERSION}\"|" "${PYPROJECT_FILE}"

# __init__.py: __version__ = "x.y.z"
if ! grep -qE '__version__[[:space:]]*=' "${INIT_FILE}"; then
  fatal "${INIT_FILE} 中未找到 __version__ 字段"
fi
sed_inplace -E "s|__version__[[:space:]]*=[[:space:]]*\".*\"|__version__ = \"${NEW_VERSION}\"|" "${INIT_FILE}"

# 校验
PYPROJECT_VERSION=$(grep -E '^version[[:space:]]*=' "${PYPROJECT_FILE}" | head -1 | sed -E 's/.*"([^"]+)".*/\1/')
INIT_VERSION=$(grep -E '__version__[[:space:]]*=' "${INIT_FILE}" | head -1 | sed -E 's/.*"([^"]+)".*/\1/')

if [[ "${PYPROJECT_VERSION}" != "${NEW_VERSION}" || "${INIT_VERSION}" != "${NEW_VERSION}" ]]; then
  fatal "版本号写入校验失败: pyproject=${PYPROJECT_VERSION}, init=${INIT_VERSION}"
fi
ok "版本号已更新: pyproject=${PYPROJECT_VERSION}, init=${INIT_VERSION}"

# ---------- 4. 提交版本号改动 ----------
info "[4/6] 提交版本号改动..."
git add "${PYPROJECT_FILE}" "${INIT_FILE}"

if [[ -z "$(git diff --cached --name-only)" ]]; then
  warn "无版本号变化（可能你已经在该版本上），跳过提交"
else
  git commit -m "chore(release): bump version to ${NEW_VERSION}"
  ok "已提交版本号变更"
fi

# ---------- 5. 推送分支 ----------
info "[5/6] 推送分支 ${CURRENT_BRANCH}..."
git push origin "${CURRENT_BRANCH}"
ok "分支已推送"

# ---------- 6. 打 tag 并推送 ----------
info "[6/6] 打 tag ${TAG} 并推送（将触发 GitHub Actions 构建）..."
git tag -a "${TAG}" -m "Release ${TAG}"
git push origin "${TAG}"

ok "==============================================="
ok " 🎉 发布完成: ${TAG}"
ok " GitHub Actions 将自动构建 Windows / macOS 版本"
ok " 查看进度: https://github.com/jiulingyun/spine-annotation-tool/actions"
ok "==============================================="
