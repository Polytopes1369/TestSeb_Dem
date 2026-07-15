#pragma once
// Minimal Wavefront OBJ writer used purely for offline visual debugging of the cluster grouping /
// QEM simplification pipeline: opening the exported file in Blender, MeshLab, etc. lets a human
// directly inspect whether simplifying each cluster group left any gap (crack) at the seams
// against its neighbors. Not part of any runtime/rendering path.

#include <filesystem>
#include <string>
#include <vector>
#include "geometry/ClusterGrouping.h"
#include "geometry/MeshSimplifier.h"

namespace geometry {

    // Writes a single SimplifiableMesh to `filePath` as one named OBJ object ("o objectName").
    // Locked vertices are additionally emitted as OBJ point elements ("p") in a companion
    // "<objectName>_locked" group, so a boundary-locked seam can be toggled/inspected separately
    // from the free-to-simplify interior in viewers that support point primitives.
    // Returns false if the file could not be opened for writing.
    bool ExportSimplifiableMeshToOBJ(
        const std::filesystem::path& filePath,
        const SimplifiableMesh& mesh,
        const std::string& objectName);

    // Writes every group's mesh into one OBJ file, each as its own named object
    // ("o group_<index>"), so the whole cluster-group simplification pass can be inspected in a
    // single load; adjacent groups sharing a locked boundary should show zero visible gap between
    // their meshes. Any pre-existing file at `filePath` is overwritten.
    // Returns false if the file could not be opened for writing.
    bool ExportClusterGroupsToOBJ(
        const std::filesystem::path& filePath,
        const std::vector<ClusterGroup>& groups);

}
