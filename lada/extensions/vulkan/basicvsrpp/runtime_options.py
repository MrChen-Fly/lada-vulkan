from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class RestorationSchedulingOptions:
    """Describe how the restoration pipeline should batch and release work."""

    stream_restore_head_chunk_size: int | None = None
    stream_restore_chunk_size: int | None = None
    detector_batch_size: int = 4
    detector_segment_length: int | None = None

    def normalized(self) -> "RestorationSchedulingOptions":
        """Return a copy with validated positive integer values."""
        head_chunk_size = self.stream_restore_head_chunk_size
        if head_chunk_size is not None:
            head_chunk_size = int(head_chunk_size)
            if head_chunk_size <= 0:
                head_chunk_size = None

        chunk_size = self.stream_restore_chunk_size
        if chunk_size is not None:
            chunk_size = int(chunk_size)
            if chunk_size <= 0:
                chunk_size = None

        detector_batch_size = max(int(self.detector_batch_size), 1)

        detector_segment_length = self.detector_segment_length
        if detector_segment_length is not None:
            detector_segment_length = int(detector_segment_length)
            if detector_segment_length <= 0:
                detector_segment_length = None

        return replace(
            self,
            stream_restore_head_chunk_size=head_chunk_size,
            stream_restore_chunk_size=chunk_size,
            detector_batch_size=detector_batch_size,
            detector_segment_length=detector_segment_length,
        )

    def resolve_detector_segment_length(self, *, max_clip_length: int) -> int:
        """Resolve the clip emission length for detector output."""
        options = self.normalized()
        if options.detector_segment_length is not None:
            return min(options.detector_segment_length, max_clip_length)
        return max_clip_length

    def resolve_frame_detection_buffer_limit(
        self,
        *,
        max_clip_length: int,
        queue_maxsize: int,
    ) -> int:
        """Resolve how many decoded frames the frame restorer should prebuffer."""
        options = self.normalized()
        preferred = options.stream_restore_chunk_size or self.resolve_detector_segment_length(
            max_clip_length=max_clip_length,
        )
        if options.stream_restore_head_chunk_size is not None:
            preferred = min(preferred, options.stream_restore_head_chunk_size)
        preferred = max(int(preferred), 8)
        if queue_maxsize <= 0:
            return preferred
        return min(int(queue_maxsize), preferred)


@dataclass(frozen=True)
class RestorationRuntimeFeatures:
    """Describe optional runtime fast paths exposed by one restoration backend."""

    use_native_blob_bridge: bool = False
    use_native_frame_preprocess: bool = False
    use_native_frame_preprocess_batch: bool = False
    use_native_fused_restore_clip: bool = False
    use_native_recurrent_runtime: bool = False
    use_native_clip_runner: bool = False
    supports_descriptor_restore: bool = False
    use_native_blend_patch: bool = False
    use_native_blend_patch_inplace: bool = False
    use_native_padded_blend_patch: bool = False
    use_native_padded_blend_patch_inplace: bool = False
    use_native_padded_blend_patch_batch_inplace: bool = False


def resolve_restoration_scheduling_options(
    model: object,
) -> RestorationSchedulingOptions:
    """Read restoration scheduling knobs from a model in one place."""
    get_options = getattr(model, "get_runtime_scheduling_options", None)
    if callable(get_options):
        options = get_options()
        if isinstance(options, RestorationSchedulingOptions):
            return options.normalized()

    return RestorationSchedulingOptions(
        stream_restore_head_chunk_size=getattr(
            model, "stream_restore_head_chunk_size", None
        ),
        stream_restore_chunk_size=getattr(model, "stream_restore_chunk_size", None),
        detector_batch_size=int(getattr(model, "detector_batch_size", 4)),
        detector_segment_length=getattr(model, "detector_segment_length", None),
    ).normalized()


def resolve_restoration_runtime_features(
    model: object,
) -> RestorationRuntimeFeatures:
    """Read backend runtime feature flags from a model in one place."""
    get_features = getattr(model, "get_runtime_features", None)
    if callable(get_features):
        features = get_features()
        if isinstance(features, RestorationRuntimeFeatures):
            return features

    return RestorationRuntimeFeatures(
        use_native_blob_bridge=bool(
            getattr(model, "use_native_blob_bridge", getattr(model, "use_gpu_blob_bridge", False))
        ),
        use_native_frame_preprocess=bool(
            getattr(
                model,
                "use_native_frame_preprocess",
                getattr(model, "use_vulkan_frame_preprocess", False),
            )
        ),
        use_native_frame_preprocess_batch=bool(
            getattr(
                model,
                "use_native_frame_preprocess_batch",
                getattr(model, "use_vulkan_frame_preprocess_batch", False),
            )
        ),
        use_native_fused_restore_clip=bool(
            getattr(
                model,
                "use_native_fused_restore_clip",
                getattr(model, "use_fused_restore_clip", False),
            )
        ),
        use_native_recurrent_runtime=bool(
            getattr(
                model,
                "use_native_recurrent_runtime",
                getattr(model, "use_recurrent_modular_runtime", False),
            )
        ),
        use_native_clip_runner=bool(getattr(model, "use_native_clip_runner", False)),
        supports_descriptor_restore=bool(
            getattr(model, "supports_descriptor_restore", False)
        ),
        use_native_blend_patch=bool(
            getattr(
                model,
                "use_native_blend_patch",
                getattr(model, "use_vulkan_blend_patch", False),
            )
        ),
        use_native_blend_patch_inplace=bool(
            getattr(
                model,
                "use_native_blend_patch_inplace",
                getattr(model, "use_vulkan_blend_patch_inplace", False),
            )
        ),
        use_native_padded_blend_patch=bool(
            getattr(
                model,
                "use_native_padded_blend_patch",
                getattr(model, "use_vulkan_blend_patch_preprocess", False),
            )
        ),
        use_native_padded_blend_patch_inplace=bool(
            getattr(
                model,
                "use_native_padded_blend_patch_inplace",
                getattr(model, "use_vulkan_blend_patch_preprocess_inplace", False),
            )
        ),
        use_native_padded_blend_patch_batch_inplace=bool(
            getattr(
                model,
                "use_native_padded_blend_patch_batch_inplace",
                getattr(model, "use_vulkan_blend_patch_preprocess_batch_inplace", False),
            )
        ),
    )
