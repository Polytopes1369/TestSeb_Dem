#ifndef GEOMETRY_PAGE_TABLE_GLSL
#define GEOMETRY_PAGE_TABLE_GLSL

// GPU-resident mirror of geometry::GpuPageTable / renderer::GpuGeometryPagePool
// (src/geometry/GpuPageTable.h, src/renderer/GpuGeometryPagePool.h): one uint32 entry per
// logical page ID, holding either the physical page slot index that page currently resolves to
// inside the physical pool buffer, or PAGE_TABLE_UNMAPPED if the page is not resident. A logical
// page ID is geometry::GpuPageTable::LogicalAddressToPageID(clusterVirtualAddress) -- the CPU
// divides the (page-aligned) cluster virtual address by GEOMETRY_PAGE_SIZE_BYTES once, up front,
// so this shader never has to perform that division itself.
//
// The includer must #define PAGE_TABLE_SET / PAGE_TABLE_BINDING to the descriptor set/binding
// this buffer is bound to before including this header (matching
// renderer::GpuGeometryPagePool::GetPageTableBuffer()'s binding in whatever descriptor set layout
// the includer's pipeline uses).

#define PAGE_TABLE_UNMAPPED 0xFFFFFFFFu
#define GEOMETRY_PAGE_SIZE_BYTES 4096u

layout(std430, set = PAGE_TABLE_SET, binding = PAGE_TABLE_BINDING) readonly buffer GeometryPageTableSSBO {
    uint physicalPageIndex[];
} g_GeometryPageTable;

bool IsClusterResident(uint logicalPageID) {
    return g_GeometryPageTable.physicalPageIndex[logicalPageID] != PAGE_TABLE_UNMAPPED;
}

// Byte offset into the physical pool buffer (renderer::GpuGeometryPagePool::GetPhysicalPoolBuffer())
// for a resident page. Only meaningful when IsClusterResident(logicalPageID) is true.
uint GetClusterPhysicalByteOffset(uint logicalPageID) {
    return g_GeometryPageTable.physicalPageIndex[logicalPageID] * GEOMETRY_PAGE_SIZE_BYTES;
}

#endif // GEOMETRY_PAGE_TABLE_GLSL
