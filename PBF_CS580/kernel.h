#ifndef kernel_h
#define kernel_h

#include "kernel.cuh"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"
typedef glm::vec3 point_t;
typedef glm::vec3 vec_t;
typedef float3 d_point_t;
typedef float3 d_vec_t;
float Poly6Value(const point_t &r, const float h);

vec_t SpikyGradient(const point_t &r, const float h);
 // namespace pbf

#endif // kernel_h
