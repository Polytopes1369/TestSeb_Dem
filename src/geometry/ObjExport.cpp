#include "geometry/ObjExport.h"

#include <fstream>

namespace geometry {

    namespace {

        // Writes one mesh's `v` lines followed by its `f` lines (and, if any vertex is locked,
        // a companion "<objectName>_locked" object made of `p` point elements), using
        // `vertexIndexBase` as the OBJ 1-based index of this mesh's first vertex -- OBJ indices
        // are global to the whole file, so callers writing multiple meshes into one file must
        // track and advance this base themselves.
        void WriteMeshObject(std::ofstream& out, const SimplifiableMesh& mesh, const std::string& objectName, uint32_t vertexIndexBase) {
            out << "o " << objectName << "\n";
            for (const maths::vec3& p : mesh.positions) {
                out << "v " << p.x << " " << p.y << " " << p.z << "\n";
            }

            out << "g " << objectName << "\n";
            for (size_t t = 0; t + 2 < mesh.triangles.size(); t += 3) {
                uint32_t i0 = vertexIndexBase + mesh.triangles[t + 0] + 1u; // OBJ indices are 1-based.
                uint32_t i1 = vertexIndexBase + mesh.triangles[t + 1] + 1u;
                uint32_t i2 = vertexIndexBase + mesh.triangles[t + 2] + 1u;
                out << "f " << i0 << " " << i1 << " " << i2 << "\n";
            }

            bool hasLocked = false;
            for (bool locked : mesh.locked) {
                if (locked) { hasLocked = true; break; }
            }
            if (hasLocked) {
                out << "g " << objectName << "_locked\n";
                for (size_t v = 0; v < mesh.locked.size(); ++v) {
                    if (mesh.locked[v]) {
                        out << "p " << (vertexIndexBase + static_cast<uint32_t>(v) + 1u) << "\n";
                    }
                }
            }
        }

    } // namespace

    bool ExportSimplifiableMeshToOBJ(const std::filesystem::path& filePath, const SimplifiableMesh& mesh, const std::string& objectName) {
        std::ofstream out(filePath, std::ios::out | std::ios::trunc);
        if (!out.is_open()) {
            return false;
        }

        out << "# " << objectName << " -- " << (mesh.triangles.size() / 3) << " triangles, "
            << mesh.positions.size() << " vertices\n";
        WriteMeshObject(out, mesh, objectName, 0u);
        return true;
    }

    bool ExportClusterGroupsToOBJ(const std::filesystem::path& filePath, const std::vector<ClusterGroup>& groups) {
        std::ofstream out(filePath, std::ios::out | std::ios::trunc);
        if (!out.is_open()) {
            return false;
        }

        out << "# Cluster group simplification debug export -- " << groups.size() << " group(s)\n";
        out << "# Each 'o group_<i>' is one simplified cluster group; 'group_<i>_locked' marks its\n";
        out << "# boundary-locked vertices (points) shared with neighboring, non-merged clusters.\n";
        out << "# Adjacent groups should show no visible gap along their shared, locked boundary.\n";

        uint32_t vertexIndexBase = 0;
        for (size_t g = 0; g < groups.size(); ++g) {
            WriteMeshObject(out, groups[g].mesh, "group_" + std::to_string(g), vertexIndexBase);
            vertexIndexBase += static_cast<uint32_t>(groups[g].mesh.positions.size());
        }
        return true;
    }

}
