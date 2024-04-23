#ifndef ESDF_MAP_HPP_
#define ESDF_MAP_HPP_

#include <Eigen/Eigen>
#include <algorithm>
#include <iostream>
#include <queue>
#include <unordered_map>
#include <vector>

namespace fiesta {

const static int num_dirs_ = 12;  // faces 2 steps
const Eigen::Vector2i dirs_[num_dirs_] = {
    Eigen::Vector2i(-1, 0),  Eigen::Vector2i(1, 0),
    Eigen::Vector2i(0, -1),  Eigen::Vector2i(0, 1),

    Eigen::Vector2i(-1, -1), Eigen::Vector2i(1, 1),
    Eigen::Vector2i(-1, 1),  Eigen::Vector2i(1, -1),

    Eigen::Vector2i(-2, 0),  Eigen::Vector2i(2, 0),
    Eigen::Vector2i(0, -2),  Eigen::Vector2i(0, 2)};

#ifdef HASH_TABLE
// Eigen::Matrix hashing function
template <typename transform_>
struct MatrixHash : std::unary_function<transform_, size_t> {
    std::size_t operator()(transform_ const &matrix) const {
        // Note that it is oblivious to the storage order of Eigen matrix
        // (column- or row-major). It will give you the same hash value for two
        // different matrices if they are the transpose of each other in
        // different storage order.
        size_t seed = 0;
        for (size_t i = 0; i < matrix.size(); ++i) {
            auto elem = *(matrix.data() + i);
            seed ^= std::hash<typename transform_::Scalar>()(elem) +
                    0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};
#endif
class ESDFMap {
    // Type of queue element to be used in priority queue
    struct QueueElement {
        Eigen::Vector2i point_;
        double distance_;
        bool operator<(const QueueElement &element) const {
            return distance_ > element.distance_;
        }
    };

 public:
    // parameters & method for occupancy information updating
    double prob_hit_log_, prob_miss_log_, clamp_min_log_, clamp_max_log_,
        min_occupancy_log_;
    const double Logit(const double &x) const;
    bool Exist(const int &idx);
    double Dist(Eigen::Vector2i a, Eigen::Vector2i b);

    // parameters & methods for conversion between Pos, Vox & Idx
    bool PosInMap(Eigen::Vector2d pos);
    bool VoxInRange(Eigen::Vector2i vox, bool current_vec = true);
    void Vox2Pos(Eigen::Vector2i vox, Eigen::Vector2d &pos);
    int Vox2Idx(Eigen::Vector2i vox);
    int Vox2Idx(Eigen::Vector2i vox, int sub_sampling_factor);
    void Pos2Vox(Eigen::Vector2d pos, Eigen::Vector2i &vox);
    Eigen::Vector2i Idx2Vox(int idx);

    // HASH TABLE related
#ifdef HASH_TABLE
    // Increase the capacity from old_size to new_size, and change the old_size
    // to new_size.
    void IncreaseCapacity(int &old_size, int new_size);
    int FindAndInsert(Eigen::Vector2i hash_key);
    std::unordered_map<Eigen::Vector2i, int, MatrixHash<Eigen::Vector2i>>
        hash_table_;
    int count, reserve_size_;
#ifdef BLOCK
    int block_, block_size_, block_bit_;
#endif
    std::vector<Eigen::Vector2i> vox_buffer_;
#else  // #ifndef HASH_TABLE
    Eigen::Vector2d map_size_;
    Eigen::Vector2d min_range_, max_range_;  // map range in pos
    Eigen::Vector2i grid_size_;              // map range in index
#endif

// data are saved in vector
#ifdef PROBABILISTIC
    std::vector<double> occupancy_buffer_;  // 0 is free, 1 is occupied
#else
    std::vector<unsigned char> occupancy_buffer_;  // 0 is free, 1 is occupied
#endif
    std::vector<double> distance_buffer_;
    std::vector<int> num_hit_, num_miss_;
    std::vector<Eigen::Vector2i> closest_obstacle_;
    std::vector<int> head_, prev_, next_;

    std::queue<QueueElement> insert_queue_;
    std::queue<QueueElement> delete_queue_;
    std::queue<QueueElement> update_queue_;
    std::queue<QueueElement> occupancy_queue_;

    // Map Properties
    Eigen::Vector2d origin_;
    int reserved_idx_4_undefined_;
    int total_time_ = 0;
    int infinity_, undefined_;
    double resolution_, resolution_inv_;
    Eigen::Vector2i max_vec_, min_vec_, last_max_vec_, last_min_vec_;

    // DLL Operations
    void DeleteFromList(int link, int idx);
    void InsertIntoList(int link, int idx);

 public:
#ifdef HASH_TABLE
    ESDFMap(Eigen::Vector2d origin, double resolution, int reserve_size = 0);
#else
    int grid_total_size_;
    ESDFMap(Eigen::Vector2d origin, double resolution,
            Eigen::Vector2d map_size);
#endif

    ~ESDFMap() {
        // TODO: implement this
    }

#ifdef PROBABILISTIC
    void SetParameters(double p_hit, double p_miss, double p_min, double p_max,
                       double p_occ);
#endif

    bool CheckUpdate();
    bool UpdateOccupancy(bool global_map);
    void UpdateESDF();

    // Occupancy Management
    int SetOccupancy(Eigen::Vector2d pos, int occ);
    int SetOccupancy(Eigen::Vector2i vox, int occ);
    int GetOccupancy(Eigen::Vector2d pos);
    int GetOccupancy(Eigen::Vector2i pos_id);

    // Distance Field Management
    double GetDistance(Eigen::Vector2d pos);
    double GetDistance(Eigen::Vector2i vox);
    double GetDistWithGradTrilinear(Eigen::Vector2d pos, Eigen::Vector2d &grad);

    // Visualization
#ifdef VISUALIZATION
    void GetPointCloud(sensor_msgs::PointCloud &m, int vis_lower_bound,
                       int vis_upper_bound);
    void GetSliceMarker(visualization_msgs::Marker &m, int slice, int id,
                        Eigen::Vector4d color, double max_dist);
#endif

    // Local Range
    void SetUpdateRange(Eigen::Vector2d min_pos, Eigen::Vector2d max_pos,
                        bool new_vec = true);
    void SetOriginalRange();

#ifndef PROBABILISTIC
    // For Deterministic Occupancy Grid
    void SetAway();
    void SetAway(Eigen::Vector2i left, Eigen::Vector2i right);
    void SetBack();
    void SetBack(Eigen::Vector2i left, Eigen::Vector2i right);
#endif

#ifdef DEBUG
    // only for test, check whether consistent
    bool CheckConsistency();
    // only for test, check between Ground Truth calculated by k-d tree
    bool CheckWithGroundTruth();
#endif
};
}  // namespace fiesta

#endif  // ESDF_MAP_HPP_
