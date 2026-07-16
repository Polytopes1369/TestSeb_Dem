#ifndef MATH_UTILS_GLSL
#define MATH_UTILS_GLSL

// Enable 64-bit integer support in GLSL
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#define saturate(x) clamp(x, 0., 1.)

//-----------------------------------------------------------//
// ------------------- CONFIG and CONSTS ------------------- //
//-----------------------------------------------------------//
const float PI = 3.14159265358979323846;
const float TAU = 6.28318530717958647692;
const float HALF_PI = 1.57079632679489661923;
const float GOLDEN_ANGLE = 2.39996323414995079549; // in rad: PI * (3. - sqrt(5.))
const float INVERSE_GOLDEN_RATIO = 6.1803398874989484820; // (sqrt(5.) - 1.) * .5;
const float EPSILON = 1e-6; // .000001
const float FARCLIP = 5e3; // 5000;

//-----------------------------------------------------------//
// -------------------- MATH FUNCTIONS --------------------- //
//-----------------------------------------------------------//
vec3 SafeNormalize(vec3 v)
{
	float dp3 = dot(v, v);

	return (dp3 > 1e-6) ? v * inversesqrt(dp3) : vec3(0., 1., 0.);
}

float EaseOutQuad(const in float x)
{
	return (-x * (x - 2.));
}

float P2(float value)
{
	return value * value;
}

vec2 P2(vec2 value)
{
	return value * value;
}

vec3 P2(vec3 value)
{
	return value * value;
}

float P3(float value)
{
	return value * value * value;
}

vec2 P3(vec2 value)
{
	return value * value * value;
}

vec3 P3(vec3 value)
{
	return value * value * value;
}

float P4(float value) 
{ 
    float v2 = value * value;
    return v2 * v2;
}

vec2 P4(vec2 value)
{ 
    vec2 v2 = value * value;
    return v2 * v2;
}

vec3 P4(vec3 value)
{ 
    vec3 v2 = value * value;
    return v2 * v2;
}

float FastExpNeg(float k, float x) 
{
    float kx = k * x;
    return 1. / (1. + kx + .48 * kx * kx);
}

// Hash function for procedural generation
uint64_t hash(uint64_t x, uint64_t seed) {
    x = (x ^ seed) * 0x9E3779B97F4A7C15UL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9UL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBUL;
    return x ^ (x >> 31);
}

// Convert hash to float
float hashFloat(uint64_t x, uint64_t seed) {
    return float(hash(x, seed) & 0xFFFFFFFFU) / 4294967295.0;
}

float Hash(vec2 p2)
{
    vec3 p3  = fract(vec3(p2.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float Hash(vec3 p3)
{
    p3  = fract(p3 * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

vec2 Hash2(vec2 p2)
{
    vec3 p3 = fract(vec3(p2.xyx) * vec3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

vec4 Hash4(vec2 p)
{
    vec4 p4 = fract(vec4(p.xyxy) * vec4(.1031, .1030, .0973, .1099));
    p4 += dot(p4, p4.wzxy + 33.33);
    return fract((p4.xxyz + p4.yzzw) * p4.zywx);
}

// Van der Corput radical inverse in the given prime base -- the low-discrepancy sequence
// building block for Halton23 below.
float RadicalInverse(uint index, uint base) {
    float result = 0.0;
    float f = 1.0 / float(base);
    uint i = index;
    while (i > 0u) {
        result += f * float(i % base);
        i /= base;
        f /= float(base);
    }
    return result;
}

// 2D Halton sequence (bases 2, 3) -- a low-discrepancy alternative to per-pixel random jitter,
// used to decorrelate successive frames' ray/sample directions without clustering.
vec2 Halton23(uint index) {
    return vec2(RadicalInverse(index, 2u), RadicalInverse(index, 3u));
}
#endif