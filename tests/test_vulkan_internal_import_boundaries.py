import unittest
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_INTERNAL_EXTENSION_CONSUMERS = {
    "lada/parity.py": (
        "lada.restorationpipeline.basicvsrpp_vulkan_restorer",
    ),
    "lada/parity_restoration_core.py": (
        "lada.restorationpipeline.basicvsrpp_vulkan_io",
        "lada.restorationpipeline.basicvsrpp_vulkan_restorer",
    ),
    "lada/parity_restoration_pipeline.py": (
        "lada.restorationpipeline.basicvsrpp_vulkan_restore_paths",
    ),
    "lada/parity_restoration_rollout.py": (
        "lada.restorationpipeline.basicvsrpp_vulkan_io",
        "lada.restorationpipeline.basicvsrpp_vulkan_recurrent_runtime",
        "lada.restorationpipeline.basicvsrpp_vulkan_restorer",
    ),
    "lada/extensions/vulkan/basicvsrpp_runtime_bootstrap.py": (
        "lada.restorationpipeline.basicvsrpp_vulkan_restorer",
    ),
}


class VulkanInternalImportBoundaryTests(unittest.TestCase):
    def test_internal_modules_do_not_import_compatibility_shims(self):
        violations: list[str] = []

        for relative_path, banned_imports in _INTERNAL_EXTENSION_CONSUMERS.items():
            source = (_ROOT / relative_path).read_text(encoding="utf-8")
            for banned_import in banned_imports:
                if banned_import in source:
                    violations.append(f"{relative_path}: {banned_import}")

        self.assertEqual(
            [],
            violations,
            "Internal Vulkan tooling should import extension modules directly, "
            f"found compatibility-shim dependencies: {violations}",
        )


if __name__ == "__main__":
    unittest.main()
