//------------------------------------------------------------------------------
// VERSION 0.9.1
//
// LICENSE
//   This software is dual-licensed to the public domain and under the following
//   license: you are granted a perpetual, irrevocable license to copy, modify,
//   publish, and distribute this file as you see fit.
//
// CREDITS
//   Written by Michal Cichon
//------------------------------------------------------------------------------
# ifndef __IMGUI_EXTRA_MATH_H__
# define __IMGUI_EXTRA_MATH_H__
# pragma once


//------------------------------------------------------------------------------
# ifndef IMGUI_DEFINE_MATH_OPERATORS
#     define IMGUI_DEFINE_MATH_OPERATORS
# endif
# include <imgui.h>
# include <imgui_internal.h>


//------------------------------------------------------------------------------
struct ImLine
{
    ImVec2 A, B;
};


//------------------------------------------------------------------------------
# if IMGUI_VERSION_NUM < 19002
inline bool operator==(const ImVec2& lhs, const ImVec2& rhs);
inline bool operator!=(const ImVec2& lhs, const ImVec2& rhs);
# endif
// LOCAL COMPATIBILITY PATCH (DemoSceneVK, Phase 7.1 PCG editor-tooling roadmap): upstream
// (thedmd/imgui-node-editor, master commit 021aa0ea4da13fed864bafb2a92d4c5205076866) left this
// declaration unguarded, unlike its own operator==/operator!= siblings directly above. Dear
// ImGui's own imgui.h (under IMGUI_DEFINE_MATH_OPERATORS, which this header forces on just above)
// has provided `operator*(const float, const ImVec2&)` since the same ImGui revision that added
// the unconditional operator==/!= this file already version-gates -- so the exact same
// `IMGUI_VERSION_NUM < 19002` guard applies here too. Without it, this declaration's matching
// definition in imgui_extra_math.inl collides with imgui.h's own (duplicate function body, a hard
// compile error) against this project's vendored ImGui 1.92.9 WIP (docking) -- verified failing
// without this guard, verified building clean with it, both on this exact vendored ImGui version.
# if IMGUI_VERSION_NUM < 19002
inline ImVec2 operator*(const float lhs, const ImVec2& rhs);
# endif
# if IMGUI_VERSION_NUM < 18955
inline ImVec2 operator-(const ImVec2& lhs);
# endif


//------------------------------------------------------------------------------
inline float  ImLength(float v);
inline float  ImLength(const ImVec2& v);
inline float  ImLengthSqr(float v);
inline ImVec2 ImNormalized(const ImVec2& v);


//------------------------------------------------------------------------------
inline bool   ImRect_IsEmpty(const ImRect& rect);
inline ImVec2 ImRect_ClosestPoint(const ImRect& rect, const ImVec2& p, bool snap_to_edge);
inline ImVec2 ImRect_ClosestPoint(const ImRect& rect, const ImVec2& p, bool snap_to_edge, float radius);
inline ImVec2 ImRect_ClosestPoint(const ImRect& rect, const ImRect& b);
inline ImLine ImRect_ClosestLine(const ImRect& rect_a, const ImRect& rect_b);
inline ImLine ImRect_ClosestLine(const ImRect& rect_a, const ImRect& rect_b, float radius_a, float radius_b);



//------------------------------------------------------------------------------
namespace ImEasing {

template <typename V, typename T>
inline V EaseOutQuad(V b, V c, T t)
{
    return b - c * (t * (t - 2));
}

} // namespace ImEasing


//------------------------------------------------------------------------------
# include "imgui_extra_math.inl"


//------------------------------------------------------------------------------
# endif // __IMGUI_EXTRA_MATH_H__
