from __future__ import annotations


_DEFAULT_CLIP_SIZE = 256
_BASICVSRPP_CLIP_SIZE_OPTIONS = (256, 320, 384)


def resolve_restoration_clip_size_options(
    model_name: str,
) -> tuple[int, ...]:
    """Return the allowed clip sizes for one Vulkan restoration backend."""
    if str(model_name).startswith("basicvsrpp"):
        return _BASICVSRPP_CLIP_SIZE_OPTIONS
    return (_DEFAULT_CLIP_SIZE,)


def resolve_clip_size_for_edge(
    max_crop_edge: int,
    clip_sizes: int | tuple[int, ...],
) -> int:
    """Pick the smallest allowed clip size that can hold one crop edge."""
    if isinstance(clip_sizes, int):
        return max(int(clip_sizes), 1)

    normalized_sizes = tuple(
        sorted(max(int(size), 1) for size in clip_sizes)
    )
    if not normalized_sizes:
        return _DEFAULT_CLIP_SIZE

    edge = max(int(max_crop_edge), 1)
    for size in normalized_sizes:
        if edge <= size:
            return size
    return normalized_sizes[-1]


def resolve_max_clip_size(clip_sizes: int | tuple[int, ...]) -> int:
    """Return the largest clip size that the current policy can emit."""
    if isinstance(clip_sizes, int):
        return max(int(clip_sizes), 1)
    if not clip_sizes:
        return _DEFAULT_CLIP_SIZE
    return max(max(int(size), 1) for size in clip_sizes)
