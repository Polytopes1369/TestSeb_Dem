#ifndef CLUSTER_VERTEX_DECODE_GLSL
#define CLUSTER_VERTEX_DECODE_GLSL

// GPU-side mirror of geometry::GeometryEncoding's CPU decode functions (see
// src/geometry/GeometryEncoding.h) for the compact on-disk vertex format written into
// geometry::ClusterData (see ClusterFormat.h): every function below must reproduce its CPU
// counterpart bit-for-bit, since the CPU is what actually wrote the bytes a resident cluster's
// physical page holds -- any divergence here would silently reconstruct the wrong geometry.
//
// The includer must #define COMPRESSED_POOL_SET / COMPRESSED_POOL_BINDING before including this
// header. The compressed pool is declared as a flat readonly uint[] (not a typed struct) because
// geometry::ClusterData is a `#pragma pack(push, 1)` C++ struct with sub-4-byte-aligned fields (a
// 3-byte-packed octahedral normal, back-to-back uint16 channels) that GLSL's std430 layout cannot
// express directly -- every field is instead reconstructed with explicit byte-offset loads, which
// is correct regardless of the field's alignment relative to a 4-byte SSBO word.

// Matches geometry::kMaxClusterVertices (ClusterFormat.h) and geometry::kPageSizeBytes.
#define CLUSTER_MAX_VERTICES 64u
#define CLUSTER_PAGE_SIZE_BYTES 4096u

// Byte layout of geometry::ClusterData, in field declaration order (positions, then normals,
// then UVs, then the local triangle-list indices this header does not need to decode):
//   ClusterVertexPosition positions[64]; //  6 bytes/vertex -> 384 bytes, offset   0
//   ClusterVertexNormal   normals[64];   //  3 bytes/vertex -> 192 bytes, offset 384
//   ClusterVertexUV       uvs[64];       //  4 bytes/vertex -> 256 bytes, offset 576
//   uint8_t               indices[384];  //  1 byte/index   -> 384 bytes, offset 832 (not decoded here)
#define CLUSTER_POSITION_BLOCK_BYTES (CLUSTER_MAX_VERTICES * 6u)
#define CLUSTER_NORMAL_BLOCK_BYTES   (CLUSTER_MAX_VERTICES * 3u)
#define CLUSTER_POSITION_BASE_BYTES  0u
#define CLUSTER_NORMAL_BASE_BYTES    (CLUSTER_POSITION_BASE_BYTES + CLUSTER_POSITION_BLOCK_BYTES)
#define CLUSTER_UV_BASE_BYTES        (CLUSTER_NORMAL_BASE_BYTES + CLUSTER_NORMAL_BLOCK_BYTES)

layout(std430, set = COMPRESSED_POOL_SET, binding = COMPRESSED_POOL_BINDING) readonly buffer CompressedClusterPoolSSBO {
    uint words[];
} g_CompressedClusterPool;

uint ClusterLoadByte(uint byteOffset) {
    uint word = g_CompressedClusterPool.words[byteOffset >> 2];
    uint shift = (byteOffset & 3u) * 8u;
    return (word >> shift) & 0xFFu;
}

uint ClusterLoadUShort(uint byteOffset) {
    // Never assumed to land on a word boundary -- assembled from two independent byte loads so
    // this is correct for any byteOffset, matching how ClusterVertexPosition's #pragma pack(1)
    // fields land at odd byte offsets for roughly half of a cluster's vertices.
    uint lo = ClusterLoadByte(byteOffset);
    uint hi = ClusterLoadByte(byteOffset + 1u);
    return lo | (hi << 8u);
}

// Vertex position: geometry::GeometryEncoding::DequantizePosition(), bit-for-bit. `boundsMin`/
// `boundsMax` must be the exact same per-cluster AABB (geometry::ClusterIndexEntry::boundsMin/
// boundsMax) the CPU used to quantize this cluster's positions in the first place.
vec3 DecodeClusterPosition(uint pageByteBase, uint vertexIndex, vec3 boundsMin, vec3 boundsMax) {
    uint byteOffset = pageByteBase + CLUSTER_POSITION_BASE_BYTES + vertexIndex * 6u;
    uint qx = ClusterLoadUShort(byteOffset + 0u);
    uint qy = ClusterLoadUShort(byteOffset + 2u);
    uint qz = ClusterLoadUShort(byteOffset + 4u);
    vec3 t = vec3(qx, qy, qz) / 65535.0;
    return boundsMin + t * (boundsMax - boundsMin);
}

// Vertex UV: geometry::GeometryEncoding::DecodeUV(), bit-for-bit. CLUSTER_UV_BASE_BYTES +
// vertexIndex*4 is always a multiple of 4 (both terms are), so the packed (u, v) half-float pair
// can be loaded as a single aligned word and unpacked with the hardware-native unpackHalf2x16
// instead of a manual byte-by-byte half-float decode.
vec2 DecodeClusterUV(uint pageByteBase, uint vertexIndex) {
    uint byteOffset = pageByteBase + CLUSTER_UV_BASE_BYTES + vertexIndex * 4u;
    uint packed = g_CompressedClusterPool.words[byteOffset >> 2];
    return unpackHalf2x16(packed);
}

// Vertex normal: geometry::GeometryEncoding::DecodeOctNormal24(), bit-for-bit (Cigolle et al.
// octahedral mapping, two 12-bit channels packed across 3 bytes).
vec3 DecodeClusterNormal(uint pageByteBase, uint vertexIndex) {
    uint byteOffset = pageByteBase + CLUSTER_NORMAL_BASE_BYTES + vertexIndex * 3u;
    uint data0 = ClusterLoadByte(byteOffset + 0u);
    uint data1 = ClusterLoadByte(byteOffset + 1u);
    uint data2 = ClusterLoadByte(byteOffset + 2u);

    uint u = data0 | ((data1 & 0x0Fu) << 8u);
    uint v = ((data1 >> 4u) & 0x0Fu) | (data2 << 4u);

    const float kChannelMax = 4095.0;
    float px = (float(u) / kChannelMax) * 2.0 - 1.0;
    float py = (float(v) / kChannelMax) * 2.0 - 1.0;

    vec3 n = vec3(px, py, 1.0 - abs(px) - abs(py));
    if (n.z < 0.0) {
        float oldX = n.x;
        n.x = (1.0 - abs(n.y)) * ((oldX >= 0.0) ? 1.0 : -1.0);
        n.y = (1.0 - abs(oldX)) * ((n.y >= 0.0) ? 1.0 : -1.0);
    }
    return normalize(n);
}

#endif // CLUSTER_VERTEX_DECODE_GLSL
