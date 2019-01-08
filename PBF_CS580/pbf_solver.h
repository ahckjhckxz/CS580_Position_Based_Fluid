#ifndef pbf_solver_h
#define pbf_solver_h

#include <unordered_set>
#include <vector>
#include <glm/glm.hpp>
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"

typedef glm::vec3 point_t;
typedef glm::vec3 vec_t;
typedef float3 d_point_t;
typedef float3 d_vec_t;
namespace pbf {
class PbfSolver {
protected:
	// kernel function h
	float h_;
	// Mass of a particle. All particles have the same mass.
	float mass_;
	// Rest density of a particle.
	float rho_0_;
	// Reciprocal of rho_0_;
	float rho_0_recpr_;
	// Epsilon in Eq (11)
	float epsilon_;
	unsigned num_iters_;

	// Tensile instanbility correction
	float corr_delta_q_coeff_;
	float corr_k_;
	unsigned corr_n_;

	float vorticity_epsilon_;
	float xsph_c_;

	float world_size_x_;
	float world_size_y_;
	float world_size_z_;

	ParticleSystem *ps_;
public:
  PbfSolver() {}

  void Update(float dt);

private:
  // overrides

  // helpers to implement this solver
  void ResetParticleRecords_();

  void RecordOldPositions_();

  void ImposeBoundaryConstraint_();

  void FindNeighbors_();

  float ComputeLambda_(size_t p_i) const;

  // @p_i: index of particle i.
  float ComputeDensityConstraint_(size_t p_i) const;

  vec_t ComputeDeltaPos_(size_t p_i) const;

  float ComputScorr_(const vec_t vec_ji) const;

  vec_t ComputeVorticity_(size_t p_i) const;

  vec_t ComputeVorticityCorrForce_(size_t p_i) const;

  vec_t ComputeEta_(size_t p_i) const;

  vec_t ComputeXsph_(size_t p_i) const;

private:
  //GravityEffect gravity_{};
  //SpatialHash<size_t, PositionGetter> spatial_hash_;

  class ParticleRecord {
  public:
    void ClearNeighbors() { neighbor_idxs.clear(); }

    void AddNeighbor(size_t i) { neighbor_idxs.insert(i); }

  public:
    // std::vector<size_t> neighbor_idxs;
    std::unordered_set<size_t> neighbor_idxs;
    float lambda{0.0f};

    vec_t old_pos{0.0f};
    vec_t delta_pos{0.0f};
    vec_t vorticity{0.0f};
  };

  std::vector<ParticleRecord> ptc_records_;
};
} // namespace pbf
#endif /* pbf_solver_h */
