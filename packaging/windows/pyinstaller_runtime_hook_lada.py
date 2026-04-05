import os
import sys

_DLL_DIR_HANDLES = []
base_path = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))

paths = [base_path,os.path.join(base_path, "bin")]
local_ncnn_runtime_dir = os.path.join(
    base_path,
    "native",
    "ncnn_vulkan_runtime",
    "build",
    "local_runtime",
)
if os.path.isdir(local_ncnn_runtime_dir):
    paths.insert(0, local_ncnn_runtime_dir)
    if local_ncnn_runtime_dir not in sys.path:
        sys.path.insert(0, local_ncnn_runtime_dir)
    os.environ["LADA_LOCAL_NCNN_RUNTIME_DIR"] = local_ncnn_runtime_dir
    if hasattr(os, "add_dll_directory"):
        try:
            _DLL_DIR_HANDLES.append(os.add_dll_directory(local_ncnn_runtime_dir))
        except OSError:
            pass

intel_marker = os.path.join(base_path, "ur_adapter_level_zero.dll")
if os.path.exists(intel_marker):
    system_path = os.environ.get("PATH", "")
    if system_path:
        paths.append(system_path)

os.environ["PATH"] = os.pathsep.join(paths)
os.environ["LADA_MODEL_WEIGHTS_DIR"] = os.path.join(base_path, "model_weights")
os.environ["LOCALE_DIR"] = os.path.join(base_path, "lada", "locale")

bundled_vulkan_cache_dir = os.path.join(base_path, "vulkan_cache")
if os.path.isdir(bundled_vulkan_cache_dir):
    os.environ["LADA_BUNDLED_VULKAN_CACHE_DIR"] = bundled_vulkan_cache_dir

pnnx_exec_path = os.path.join(base_path, "pnnx", "pnnx.exe")
if os.path.isfile(pnnx_exec_path):
    os.environ["LADA_PNNX_EXEC_PATH"] = pnnx_exec_path
    try:
        import pnnx  # noqa: PLC0415

        pnnx.EXEC_PATH = pnnx_exec_path
    except Exception:
        pass
