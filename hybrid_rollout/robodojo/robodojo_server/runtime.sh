#!/usr/bin/env bash
# Source this file in the RoboDojo process; it does not modify system libraries.
export TASK_ROOT="${TASK_ROOT:-$HOME/Projects/gpt-gated-dagger}"
DAGGER_DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
export DAGGER_NVIDIA_RUNTIME="${DAGGER_NVIDIA_RUNTIME:-$HOME/.local/share/robolab-runtime/NVIDIA-Linux-x86_64-$DAGGER_DRIVER_VERSION}"
export DAGGER_SYSROOT="${DAGGER_SYSROOT:-$HOME/.local/share/robolab-runtime/sysroot}"
# Read the linker cache once: `ldconfig -p | grep -q` can SIGPIPE ldconfig and
# fail under the caller's `set -o pipefail` even when the library is present.
ld_cache="$(ldconfig -p 2>/dev/null || true)"
system_icd=''
for candidate in /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json; do
  [[ -f "$candidate" ]] && { system_icd="$candidate"; break; }
done
if [[ -f "$DAGGER_NVIDIA_RUNTIME/libGLX_nvidia.so.$DAGGER_DRIVER_VERSION" ]]; then
  # Container images without driver userspace: use the extracted private copy.
  export LD_LIBRARY_PATH="$DAGGER_NVIDIA_RUNTIME:$DAGGER_SYSROOT/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export VK_ICD_FILENAMES="$DAGGER_NVIDIA_RUNTIME/nvidia_icd.json"
  export __EGL_VENDOR_LIBRARY_FILENAMES="$DAGGER_NVIDIA_RUNTIME/10_nvidia.json"
elif [[ -n "$system_icd" ]] && [[ "$ld_cache" == *"libGLX_nvidia.so.0 "* ]] \
    && [[ -e "$(awk '/libGLX_nvidia.so.0 /{print $NF; exit}' <<< "$ld_cache")" ]]; then
  # Bare-metal/driver-injected hosts: use the installed driver, and pin a single
  # NVIDIA ICD so Mesa ICDs cannot be selected for headless RTX rendering.
  export DAGGER_NVIDIA_RUNTIME="system:$DAGGER_DRIVER_VERSION"
  export VK_ICD_FILENAMES="$system_icd"
  [[ -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]] && \
    export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
else
  printf 'Missing NVIDIA userspace graphics libraries matching driver %s\n' "$DAGGER_DRIVER_VERSION" >&2
  return 1 2>/dev/null || exit 1
fi
# Isaac Sim's MDL/iray material stack needs libGLU; without it RTX scene setup
# segfaults later with no Python traceback, so fail with the actual cause instead.
glu_found=''
[[ "$ld_cache" == *"libGLU.so.1 "* ]] && glu_found=1
IFS=: read -r -a library_dirs <<< "${LD_LIBRARY_PATH:-}"
for library_dir in "${library_dirs[@]}"; do
  [[ -n "$library_dir" && -e "$library_dir/libGLU.so.1" ]] && glu_found=1
done
if [[ -z "$glu_found" ]]; then
  printf 'Missing libGLU.so.1 (libglu1-mesa) required by Isaac Sim MDL materials on %s\n' "$(hostname)" >&2
  return 1 2>/dev/null || exit 1
fi
export XDG_RUNTIME_DIR="$TASK_ROOT/runtime/xdg"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
export OMNI_KIT_ACCEPT_EULA=Y
