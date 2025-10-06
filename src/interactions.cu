#include <thrust/random.h>

#include "interactions.h"
#include "utilities.h"

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal, thrust::default_random_engine& rng) {
  thrust::uniform_real_distribution<float> u01(0, 1);

  float up = sqrt(u01(rng));       // cos(theta)
  float over = sqrt(1 - up * up);  // sin(theta)
  float around = u01(rng) * TWO_PI;

  // Find a direction that is not the normal based off of whether or not the
  // normal's components are all equal to sqrt(1/3) or whether or not at
  // least one component is less than sqrt(1/3). Learned this trick from
  // Peter Kutz.

  glm::vec3 directionNotNormal;
  if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
    directionNotNormal = glm::vec3(1, 0, 0);
  } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
    directionNotNormal = glm::vec3(0, 1, 0);
  } else {
    directionNotNormal = glm::vec3(0, 0, 1);
  }

  // Use not-normal direction to generate two perpendicular directions
  glm::vec3 perpendicularDirection1 =
      glm::normalize(glm::cross(normal, directionNotNormal));
  glm::vec3 perpendicularDirection2 =
      glm::normalize(glm::cross(normal, perpendicularDirection1));

  return up * normal + cos(around) * over * perpendicularDirection1 +
         sin(around) * over * perpendicularDirection2;
}

/**
 * Computes the Schlick approximation for reflectance.
 */
__host__ __device__ float schlick(const float cosine,
                                  const float indexOfRefraction) {
  float r0 = (1.0f - indexOfRefraction) / (1.0f + indexOfRefraction);
  r0 = r0 * r0;
  return r0 + (1.0f - r0) * glm::pow((1.0f - cosine), 5.0f);
}

__host__ __device__ void scatterRay(PathSegment& pathSegment,
                                    glm::vec3 intersect, glm::vec3 normal,
                                    const Material& m,
                                    thrust::default_random_engine& rng) {
  // TODO: implement this.
  // A basic implementation of pure-diffuse shading will just call the
  // calculateRandomDirectionInHemisphere defined above.
  thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

  glm::vec3 newDirection;
  glm::vec3 unitIncident = glm::normalize(pathSegment.ray.direction);
  glm::vec3 n = glm::normalize(normal);

  if (m.hasReflective) {
    newDirection = glm::reflect(unitIncident, n);
  } else if (m.hasRefractive) {
    glm::vec3 unitIncident = glm::normalize(pathSegment.ray.direction);
    float cosTheta = glm::dot(-unitIncident, n);
    float ri = 1.0f / m.indexOfRefraction;
    if (cosTheta < 0.0f) {  // internal surface
      ri = m.indexOfRefraction;
      n = -n;
      cosTheta = -cosTheta;
    }
    float sinTheta = glm::sqrt(1.0f - cosTheta * cosTheta);
    bool cannot_refract = ri * sinTheta > 1.0f;
    float rand_f = u01(rng);
    float schlick_index = schlick(cosTheta, ri);
    if (cannot_refract || schlick_index > rand_f)
      newDirection = glm::reflect(unitIncident, n);
    else
      newDirection = glm::refract(unitIncident, n, ri);
  } else {
    newDirection = calculateRandomDirectionInHemisphere(normal, rng);
  }
  newDirection = glm::normalize(newDirection);
  pathSegment.ray.direction = newDirection;
  pathSegment.ray.origin = intersect + newDirection * 0.001f;
  pathSegment.color *= m.color;

#if RUSSIAN_ROULETTE
  glm::vec3& rrBeta = pathSegment.color;
  float maxComp = glm::max(rrBeta[0], glm::max(rrBeta[1], rrBeta[2]));
  if (maxComp < 1.0f && pathSegment.remainingBounces >= 1.0f) {
    if (maxComp < u01(rng)) {
      pathSegment.remainingBounces = 0.0f;
    } else {
      rrBeta /= maxComp;
    }
  }
#endif  // RUSSIAN_ROULETTE
}