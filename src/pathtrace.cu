#include <cuda.h>
#include <thrust/execution_policy.h>
#include <thrust/partition.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include <cmath>
#include <cstdio>

#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "interactions.h"
#include "intersections.h"
#include "pathtrace.h"
#include "scene.h"
#include "sceneStructs.h"
#include "utilities.h"

#define ERRORCHECK 1

#define FILENAME \
  (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line) {
#if ERRORCHECK
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess == err) {
    return;
  }

  fprintf(stderr, "CUDA error");
  if (file) {
    fprintf(stderr, " (%s:%d)", file, line);
  }
  fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
  getchar();
#endif  // _WIN32
  exit(EXIT_FAILURE);
#endif  // ERRORCHECK
}

__host__ __device__ thrust::default_random_engine makeSeededRandomEngine(
    int iter, int index, int depth) {
  int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
  return thrust::default_random_engine(h);
}

// Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter,
                               glm::vec3* image) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (x < resolution.x && y < resolution.y) {
    int index = x + (y * resolution.x);
    glm::vec3 pix = image[index];

    glm::ivec3 color;
    color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
    color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
    color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

    // Each thread writes one pixel location in the texture (textel)
    pbo[index].w = 0;
    pbo[index].x = color.x;
    pbo[index].y = color.y;
    pbo[index].z = color.z;
  }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

void InitDataContainer(GuiDataContainer* imGuiData) { guiData = imGuiData; }

void pathtraceInit(Scene* scene) {
  hst_scene = scene;

  const Camera& cam = hst_scene->state.camera;
  const int pixelcount = cam.resolution.x * cam.resolution.y;

  cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
  cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

  cudaMalloc(&dev_paths, pixelcount * SAMPLES_PER_PIXEL * sizeof(PathSegment));

  cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
  cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom),
             cudaMemcpyHostToDevice);

  cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  cudaMemcpy(dev_materials, scene->materials.data(),
             scene->materials.size() * sizeof(Material),
             cudaMemcpyHostToDevice);

  cudaMalloc(&dev_intersections,
             pixelcount * SAMPLES_PER_PIXEL * sizeof(ShadeableIntersection));
  cudaMemset(dev_intersections, 0,
             pixelcount * SAMPLES_PER_PIXEL * sizeof(ShadeableIntersection));

  // TODO: initialize any extra device memeory you need

  checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
  cudaFree(dev_image);  // no-op if dev_image is null
  cudaFree(dev_paths);
  cudaFree(dev_geoms);
  cudaFree(dev_materials);
  cudaFree(dev_intersections);
  // TODO: clean up any extra device memory you created

  checkCUDAError("pathtraceFree");
}

__device__ glm::vec2 concentricSampleDisk(float u1, float u2) {
  float r, theta;
  float sx = 2.0f * u1 - 1.0f;
  float sy = 2.0f * u2 - 1.0f;
  if (sx == 0.0f && sy == 0.0f) {
    r = 0.0f;
    theta = 0.0f;
  } else if (abs(sx) > abs(sy)) {
    r = sx;
    theta = (PI / 4.0f) * (sy / sx);
  } else {
    r = sy;
    theta = (PI / 2.0f) - (PI / 4.0f) * (sx / sy);
  }
  return glm::vec2(r * cosf(theta), r * sinf(theta));
}

/**
 * Generate PathSegments with rays from the camera through the screen into the
 * scene, which is the first bounce of rays.
 *
 * Antialiasing - add rays for sub-pixel sampling
 * motion blur - jitter rays "in time"
 * lens effect - jitter ray origin positions based on a lens
 */
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth,
                                      PathSegment* pathSegments) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (x < cam.resolution.x && y < cam.resolution.y) {
    int index = x + (y * cam.resolution.x);
    PathSegment& segment = pathSegments[index];

    segment.ray.origin = cam.position;
    segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

    // TODO: implement antialiasing by jittering the ray
#if ANTIALIASING || DEPTH_OF_FIELD
    thrust::default_random_engine rng =
        makeSeededRandomEngine(iter, index, segment.remainingBounces);
#endif

#if ANTIALIASING
    thrust::uniform_real_distribution<float> un11(-1.0f, 1.0f);
    x += un11(rng);
    y += un11(rng);
#endif  // ANTIALIASING

    glm::vec3 pixelTarget = cam.view -
                            cam.right * cam.pixelLength.x *
                                ((float)x - (float)cam.resolution.x * 0.5f) -
                            cam.up * cam.pixelLength.y *
                                ((float)y - (float)cam.resolution.y * 0.5f);

#if DEPTH_OF_FIELD
    const float lensRadius = LENS_RADIUS;
    const float focalDistance = FOCAL_Z + cam.position.z;
    thrust::uniform_real_distribution<float> u01(0, 1);
    glm::vec2 diskSample = concentricSampleDisk(u01(rng), u01(rng));
    glm::vec3 lensOffset = cam.right * diskSample.x * lensRadius +
                           cam.up * diskSample.y * lensRadius;
    glm::vec3 focusPoint =
        cam.position + glm::normalize(pixelTarget) * focalDistance;
    segment.ray.origin = cam.position + lensOffset;
    segment.ray.direction = glm::normalize(focusPoint - segment.ray.origin);
#else
    segment.ray.direction = glm::normalize(pixelTarget);
#endif

    segment.pixelIndex = index;
    segment.remainingBounces = traceDepth;
  }
}

#if MC_RANDOM_TYPE > 0

__device__ float halton(int index, int base) {
  float f = 1.0f;
  float r = 0.0f;
  while (index > 0) {
    f /= base;
    r += f * (index % base);
    index /= base;
  }
  return r;
}

__device__ float sobol(int index, int dimension) {
  unsigned int result = 0;
  for (unsigned int i = 0, mask = 1U << 31; mask; mask >>= 1, ++i) {
    if (index & mask) {
      switch (dimension) {
        case 0:
          result ^= 0x80000000U >> i;
          break;
        case 1:
          result ^= 0x40000000U >> i;
          break;
        default:
          result ^= 0x80000000U >> i;
          break;
      }
    }
  }
  return (float)result / (float)0xFFFFFFFFU;
}

__device__ float getMCFloat(int x, int y, int s, int dim,
                            thrust::default_random_engine& rng) {
#if MC_RANDOM_TYPE == 1
  thrust::uniform_real_distribution<float> u01(0, 1);
  return u01(rng);
#elif MC_RANDOM_TYPE == 2
  return (float)rand() / (float)RAND_MAX;
#elif MC_RANDOM_TYPE == 3
  int grid = 4;
  int jx = s % grid;
  int jy = s / grid;
  return ((x + (jx + 0.5f) / grid) + (y + (jy + 0.5f) / grid)) * 0.5f / grid;
#elif MC_RANDOM_TYPE == 4
  if (dim == 0)
    return halton(s + 1, 2);
  else
    return halton(s + 1, 3);
#elif MC_RANDOM_TYPE == 5
  return sobol(s + 1, dim);
#endif
}

__device__ glm::vec2 getMCSample2D(int x, int y, int s,
                                   thrust::default_random_engine& rng) {
  return glm::vec2(getMCFloat(x, y, s, 0, rng), getMCFloat(x, y, s, 1, rng));
}

__global__ void generateRayFromCameraMC(Camera cam, int iter, int traceDepth,
                                        PathSegment* pathSegments,
                                        int samplesPerPixel) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (x < cam.resolution.x && y < cam.resolution.y) {
    int index = x + (y * cam.resolution.x);
    for (int s = 0; s < samplesPerPixel; ++s) {
      PathSegment& segment = pathSegments[index * samplesPerPixel + s];
      segment.ray.origin = cam.position;
      segment.color = glm::vec3(1.0f, 1.0f, 1.0f);
      thrust::default_random_engine rng =
          makeSeededRandomEngine(iter, index, s);
      glm::vec2 jitter = getMCSample2D(x, y, s, rng);
      float fx = (float)x + jitter.x;
      float fy = (float)y + jitter.y;
      glm::vec3 pixelTarget =
          cam.view -
          cam.right * cam.pixelLength.x *
              (fx - (float)cam.resolution.x * 0.5f) -
          cam.up * cam.pixelLength.y * (fy - (float)cam.resolution.y * 0.5f);
      segment.ray.direction = glm::normalize(pixelTarget);
      segment.pixelIndex = index;
      segment.remainingBounces = traceDepth;
    }
  }
}
#endif  // MC_RANDOM_TYPE > 0

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(int depth, int num_paths,
                                     PathSegment* pathSegments, Geom* geoms,
                                     int geoms_size,
                                     ShadeableIntersection* intersections) {
  int path_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (path_index < num_paths) {
    PathSegment pathSegment = pathSegments[path_index];

    float t;
    glm::vec3 intersect_point;
    glm::vec3 normal;
    float t_min = FLT_MAX;
    int hit_geom_index = -1;
    bool outside = true;

    glm::vec3 tmp_intersect;
    glm::vec3 tmp_normal;

    // naive parse through global geoms

    for (int i = 0; i < geoms_size; i++) {
      Geom& geom = geoms[i];

      if (geom.type == CUBE) {
        t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect,
                                tmp_normal, outside);
      } else if (geom.type == SPHERE) {
        t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect,
                                   tmp_normal, outside);
      }
      // TODO: add more intersection tests here... triangle? metaball? CSG?

      // Compute the minimum t from the intersection tests to determine what
      // scene geometry object was hit first.
      if (t > 0.0f && t_min > t) {
        t_min = t;
        hit_geom_index = i;
        intersect_point = tmp_intersect;
        normal = tmp_normal;
      }
    }

    if (hit_geom_index == -1) {
      intersections[path_index].t = -1.0f;
    } else {
      // The ray hits something
      intersections[path_index].t = t_min;
      intersections[path_index].materialId = geoms[hit_geom_index].materialid;
      intersections[path_index].surfaceNormal = normal;
    }
  }
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial(int iter, int num_paths,
                                  ShadeableIntersection* shadeableIntersections,
                                  PathSegment* pathSegments,
                                  Material* materials) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths) {
    ShadeableIntersection intersection = shadeableIntersections[idx];
    if (intersection.t > 0.0f)  // if the intersection exists...
    {
      // Set up the RNG
      // LOOK: this is how you use thrust's RNG! Please look at
      // makeSeededRandomEngine as well.
      thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
      thrust::uniform_real_distribution<float> u01(0, 1);

      Material material = materials[intersection.materialId];
      glm::vec3 materialColor = material.color;

      // If the material indicates that the object was a light, "light" the ray
      if (material.emittance > 0.0f) {
        pathSegments[idx].color *= (materialColor * material.emittance);
      }
      // Otherwise, do some pseudo-lighting computation. This is actually more
      // like what you would expect from shading in a rasterizer like OpenGL.
      // TODO: replace this! you should be able to start with basically a
      // one-liner
      else {
        float lightTerm =
            glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
        pathSegments[idx].color *=
            (materialColor * lightTerm) * 0.3f +
            ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
        pathSegments[idx].color *=
            u01(rng);  // apply some noise because why not
      }
      // If there was no intersection, color the ray black.
      // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
      // used for opacity, in which case they can indicate "no opacity".
      // This can be useful for post-processing and image compositing.
    } else {
      pathSegments[idx].color = glm::vec3(0.0f);
    }
  }
}

struct isPathBouncing {
  __host__ __device__ bool operator()(const PathSegment& path) {
    return path.remainingBounces > 0;
  }
};

struct sortByMaterialId {
  __host__ __device__ bool operator()(const ShadeableIntersection& a,
                                      const ShadeableIntersection& b) {
    return a.materialId < b.materialId;
  }
};

__global__ void shadeMaterial(int iter, int num_paths,
                              ShadeableIntersection* shadeableIntersections,
                              PathSegment* pathSegments, Material* materials) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths) {
    ShadeableIntersection intersection = shadeableIntersections[idx];
    PathSegment& pathSegment = pathSegments[idx];
    if (intersection.t > 0.0f) {
      thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
      thrust::uniform_real_distribution<float> u01(0, 1);

      Material material = materials[intersection.materialId];
      glm::vec3 materialColor = material.color;

      if (material.emittance > 0.0f) {
        pathSegment.color *= (materialColor * material.emittance);
        pathSegment.remainingBounces = 0.0f;
      } else {
        scatterRay(pathSegment, getPointOnRay(pathSegment.ray, intersection.t),
                   intersection.surfaceNormal, material, rng);
        pathSegment.remainingBounces--;
      }
    } else {
      pathSegment.color = glm::vec3(0.0f);
      pathSegment.remainingBounces = 0.0f;
    }
  }
}

#if DIRECT_LIGHTING
__global__ void shadeDirectLighting(
    int iter, int num_paths, ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments, Material* materials, Geom* geoms,
    int geoms_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths) {
    ShadeableIntersection intersection = shadeableIntersections[idx];
    PathSegment& pathSegment = pathSegments[idx];
    if (intersection.t > 0.0f) {
      Material material = materials[intersection.materialId];
      if (material.emittance <= 0.0f) {
        glm::vec3 directLight(0.0f);
        for (int i = 0; i < geoms_size; ++i) {
          Material lightMat = materials[geoms[i].materialid];
          if (lightMat.emittance > 0.0f) {
            glm::vec3 lightPos, lightNormal;
            lightPos = glm::vec3(geoms[i].translation);
            lightNormal = glm::normalize(
                lightPos - getPointOnRay(pathSegment.ray, intersection.t));
            glm::vec3 hitPoint = getPointOnRay(pathSegment.ray, intersection.t);
            glm::vec3 toLight = glm::normalize(lightPos - hitPoint);
            float nDotL =
                glm::max(glm::dot(intersection.surfaceNormal, toLight), 0.0f);
            if (nDotL > 0.0f) {
              Ray shadowRay;
              shadowRay.origin = hitPoint + toLight * 0.001f;
              shadowRay.direction = toLight;
              float occluded = 0.0f;
              for (int j = 0; j < geoms_size; ++j) {
                if (j == i) continue;
                glm::vec3 tmp_intersect, tmp_normal;
                bool tmp_outside;
                float t = -1.0f;
                if (geoms[j].type == CUBE) {
                  t = boxIntersectionTest(geoms[j], shadowRay, tmp_intersect,
                                          tmp_normal, tmp_outside);
                } else if (geoms[j].type == SPHERE) {
                  t = sphereIntersectionTest(geoms[j], shadowRay, tmp_intersect,
                                             tmp_normal, tmp_outside);
                }
                float distToLight = glm::length(lightPos - hitPoint);
                if (t > 0.0f && t < distToLight - 0.01f) {
                  occluded = distToLight;
                  break;
                }
              }
              if (occluded > 0.0f) {
                directLight +=
                    lightMat.color * lightMat.emittance * nDotL * occluded;
              }
            }
          }
        }
        pathSegment.color +=
            material.color * (directLight * DIRECT_LIGHTING_SCALE);
        // normalize color components
        float maxComp =
            glm::max(pathSegment.color[0],
                     glm::max(pathSegment.color[1], pathSegment.color[2]));
        if (maxComp > 1.0f) pathSegment.color /= maxComp;
      }
    }
  }
}
#endif

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image,
                            PathSegment* iterationPaths) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (index < nPaths) {
    PathSegment iterationPath = iterationPaths[index];
    image[iterationPath.pixelIndex] += iterationPath.color;
  }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter) {
  const int traceDepth = hst_scene->state.traceDepth;
  const Camera& cam = hst_scene->state.camera;
  const int pixelcount = cam.resolution.x * cam.resolution.y;
  const int samplesPerPixel = SAMPLES_PER_PIXEL;

  // 2D block for generating ray from camera
  const dim3 blockSize2d(8, 8);
  const dim3 blocksPerGrid2d(
      (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
      (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

  // 1D block for path tracing
  const int blockSize1d = 128;

  ///////////////////////////////////////////////////////////////////////////

  // Recap:
  // * Initialize array of path rays (using rays that come out of the camera)
  //   * You can pass the Camera object to that kernel.
  //   * Each path ray must carry at minimum a (ray, color) pair,
  //   * where color starts as the multiplicative identity, white = (1, 1, 1).
  //   * This has already been done for you.
  // * For each depth:
  //   * Compute an intersection in the scene for each path ray.
  //     A very naive version of this has been implemented for you, but feel
  //     free to add more primitives and/or a better algorithm.
  //     Currently, intersection distance is recorded as a parametric distance,
  //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
  //     * Color is attenuated (multiplied) by reflections off of any object
  //   * TODO: Stream compact away all of the terminated paths.
  //     You may use either your implementation or `thrust::remove_if` or its
  //     cousins.
  //     * Note that you can't really use a 2D kernel launch any more - switch
  //       to 1D.
  //   * TODO: Shade the rays that intersected something or didn't bottom out.
  //     That is, color the ray by performing a color computation according
  //     to the shader, then generate a new ray to continue the ray path.
  //     We recommend just updating the ray's PathSegment in place.
  //     Note that this step may come before or after stream compaction,
  //     since some shaders you write may also cause a path to terminate.
  // * Finally, add this iteration's results to the image. This has been done
  //   for you.

  // TODO: perform one iteration of path tracing

#if MC_RANDOM_TYPE > 0
  generateRayFromCameraMC<<<blocksPerGrid2d, blockSize2d>>>(
      cam, iter, traceDepth, dev_paths, samplesPerPixel);
  checkCUDAError("generate camera ray MC");
  int totalPaths = pixelcount * samplesPerPixel;
#else
  generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth,
                                                          dev_paths);
  checkCUDAError("generate camera ray");
  int totalPaths = pixelcount;
#endif

  int depth = 0;
  PathSegment* dev_path_end = dev_paths + totalPaths;
  int num_paths = dev_path_end - dev_paths;

  // --- PathSegment Tracing Stage ---
  // Shoot ray into scene, bounce between objects, push shading chunks

  bool iterationComplete = false;
  while (!iterationComplete) {
    // clean shading chunks
    cudaMemset(dev_intersections, 0,
               totalPaths * sizeof(ShadeableIntersection));

    // tracing
    dim3 numblocksPathSegmentTracing =
        (num_paths + blockSize1d - 1) / blockSize1d;
    computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>>(
        depth, num_paths, dev_paths, dev_geoms, hst_scene->geoms.size(),
        dev_intersections);
    checkCUDAError("trace one bounce");
    cudaDeviceSynchronize();
    depth++;

    // TODO:
    // --- Shading Stage ---
    // Shade path segments based on intersections and generate new rays by
    // evaluating the BSDF.
    // Start off with just a big kernel that handles all the different
    // materials you have in the scenefile.
    // TODO: compare between directly shading the path segments and shading
    // path segments that have been reshuffled to be contiguous in memory.

#if SORT_MATERIALS
    thrust::sort_by_key(thrust::device, dev_intersections,
                        dev_intersections + num_paths, dev_paths,
                        sortByMaterialId());
#endif  // SORT_MATERIALS

    shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
        iter, num_paths, dev_intersections, dev_paths, dev_materials);
    checkCUDAError("shade one bounce");

#if DIRECT_LIGHTING
    shadeDirectLighting<<<numblocksPathSegmentTracing, blockSize1d>>>(
        iter, num_paths, dev_intersections, dev_paths, dev_materials, dev_geoms,
        hst_scene->geoms.size());
    checkCUDAError("shade direct lighting");
#endif

    dev_path_end = thrust::partition(thrust::device, dev_paths,
                                     dev_paths + num_paths, isPathBouncing());
    num_paths = dev_path_end - dev_paths;

    iterationComplete = (num_paths <= 0) || (depth >= traceDepth);

    if (guiData != NULL) {
      guiData->TracedDepth = depth;
    }
  }

  // Assemble this iteration and apply it to the image
  dim3 numBlocksPixels = (totalPaths + blockSize1d - 1) / blockSize1d;
  finalGather<<<numBlocksPixels, blockSize1d>>>(totalPaths, dev_image,
                                                dev_paths);

  // Send results to OpenGL buffer for rendering
  sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter,
                                                   dev_image);

  // Retrieve image from GPU
  cudaMemcpy(hst_scene->state.image.data(), dev_image,
             pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  checkCUDAError("pathtrace");
}