---
description: Memory Barriers & Synchronization
---

For any operation involving Vulkan synchronization (transitions, compute-to-raster dependency, compute-to-ray-tracing dependency), you must output an explicit layout transition block detailing:

The Pipeline Stages involved (Src & Dst).

The Access Masks (Src & Dst).

The Image Layouts (Old & New).
You must use VkBarrier2 and VkDependencyInfo layout transitions. No legacy barriers.