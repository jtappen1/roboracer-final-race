#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <nanoflann.hpp>

#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <string>

// ─────────────────────────────────────────────
//  Compile-time constants (tune these)
// ─────────────────────────────────────────────
static constexpr double WHEELBASE      = 0.15;
static constexpr double L_MIN          = 0.5;
static constexpr double L_MAX          = 3.0;
static constexpr double V_MIN          = 3;
static constexpr double V_MAX          = 5.0;
static constexpr double LOOKAHEAD_GAIN = 1.0;   // scales v_norm → lookahead
// static constexpr double MAX_STEER_RAD  = 0.698; // 40 degrees
static constexpr int    KD_CANDIDATES  = 5;      // nearest neighbours to check

// ─────────────────────────────────────────────
//  Waypoint layout: [x, y, velocity]
// ─────────────────────────────────────────────
struct Waypoint {
    float x, y, l, v;
};

// ─────────────────────────────────────────────
//  nanoflann adaptor (2-D, float)
// ─────────────────────────────────────────────
struct WaypointCloud {
    const std::vector<Waypoint>& pts;
    explicit WaypointCloud(const std::vector<Waypoint>& p) : pts(p) {}

    inline std::size_t kdtree_get_point_count() const { return pts.size(); }

    inline float kdtree_get_pt(const std::size_t idx, const std::size_t dim) const {
        return dim == 0 ? pts[idx].x : pts[idx].y;
    }

    template<class BBOX>
    bool kdtree_get_bbox(BBOX&) const { return false; }
};

using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, WaypointCloud>,
    WaypointCloud, 2 /*dims*/>;


class PurePursuit : public rclcpp::Node {
public:
    PurePursuit() : Node("final_race_pure_pursuit_node") {

        // ── load waypoints ──────────────────────────────────────
        this->declare_parameter<std::string>("waypoints_path", "/home/nvidia/ros2_ws/src/roboracer-final-race/pure_pursuit/path/reliable_race_2_wp.csv");
        const auto wp_path = this->get_parameter("waypoints_path").as_string();
        load_waypoints(wp_path);

        // ── build KD-tree once ──────────────────────────────────
        cloud_  = std::make_unique<WaypointCloud>(waypoints_);
        kdtree_ = std::make_unique<KDTree>(2, *cloud_,
                      nanoflann::KDTreeSingleIndexAdaptorParams(10 /*leaf_max_size*/));
        kdtree_->buildIndex();

        // ── ROS interfaces ──────────────────────────────────────
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "pf/pose/odom", rclcpp::SensorDataQoS(),
            [this](nav_msgs::msg::Odometry::ConstSharedPtr msg){ pose_callback(msg); });

        drive_pub_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
            "/drive", rclcpp::SystemDefaultsQoS());

        // Visualisation (low-priority, 2 Hz)
        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/final_race_pure_pursuit/waypoint_markers", rclcpp::ServicesQoS());
        vis_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(500), [this](){ publish_waypoints(); });

        goal_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/final_race_pure_pursuit/goal_marker", rclcpp::ServicesQoS());

        RCLCPP_INFO(get_logger(), "PurePursuit ready — %zu waypoints", waypoints_.size());
    }

private:
    std::vector<Waypoint> waypoints_;
    std::vector<float>    lookaheads_;

    std::unique_ptr<WaypointCloud> cloud_;
    std::unique_ptr<KDTree>        kdtree_;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr goal_pub_;
    rclcpp::TimerBase::SharedPtr vis_timer_;

    // last-known closest index (warm-start — avoids cold search every frame)
    int last_idx_ = 0;

    // ── waypoint loading ────────────────────────────────────────
    void load_waypoints(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open())
            throw std::runtime_error("Cannot open waypoints: " + path);

        waypoints_.reserve(4096);
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;  // skip comments
            if (line.empty()) continue;
            std::istringstream ss(line);
            std::string tok;
            Waypoint wp{};
            if (std::getline(ss, tok, ',')) wp.x = std::stof(tok);
            if (std::getline(ss, tok, ',')) wp.y = std::stof(tok);
            if (std::getline(ss, tok, ',')) wp.l = std::stof(tok);
            if (std::getline(ss, tok, ',')) wp.v = std::stof(tok);
            waypoints_.push_back(wp);
        }
        RCLCPP_INFO(get_logger(), "Loaded %zu waypoints from %s",
                    waypoints_.size(), path.c_str());
    }

    // ── fast nearest-waypoint (ahead of car) via KD-tree ────────
    //   Returns index into waypoints_. Uses last_idx_ as a warm-start
    //   to further cut search cost on the second pass.
    int nearest_forward(float cx, float cy, float cos_yaw, float sin_yaw) {
        // 1. KD-tree: find KD_CANDIDATES neighbours
        float query[2] = {cx, cy};
        std::array<std::size_t, KD_CANDIDATES> indices;
        std::array<float,       KD_CANDIDATES> dists_sq;

        nanoflann::KNNResultSet<float> rs(KD_CANDIDATES);
        rs.init(indices.data(), dists_sq.data());
        kdtree_->findNeighbors(rs, query, nanoflann::SearchParams());

        // 2. Among candidates, pick the closest one that is *ahead*
        int   best_idx  = -1;
        float best_dist = std::numeric_limits<float>::max();

        for (int k = 0; k < KD_CANDIDATES; ++k) {
            int i = static_cast<int>(indices[k]);
            float dx = waypoints_[i].x - cx;
            float dy = waypoints_[i].y - cy;
            // forward dot product
            float fwd = dx * cos_yaw + dy * sin_yaw;
            if (fwd > 0.0f && dists_sq[k] < best_dist) {
                best_dist = dists_sq[k];
                best_idx  = i;
            }
        }

        // 3. Fallback: just the nearest overall
        if (best_idx < 0) best_idx = static_cast<int>(indices[0]);

        last_idx_ = best_idx;
        return best_idx;
    }

    // ── lookahead point via arc-length walk ──────────────────────
    //   Walks forward from start_idx accumulating segment lengths
    //   until we reach the desired arc distance, then interpolates.
    std::array<float, 2>
    compute_lookahead_pt(int start_idx, float lookahead) const {
        const int N = static_cast<int>(waypoints_.size());
        float acc   = 0.0f;
        float px    = waypoints_[start_idx].x;
        float py    = waypoints_[start_idx].y;

        for (int i = 1; i <= N; ++i) {
            int idx = (start_idx + i) % N;
            float nx = waypoints_[idx].x;
            float ny = waypoints_[idx].y;
            float dx = nx - px;
            float dy = ny - py;
            float seg = std::sqrt(dx*dx + dy*dy);

            if (acc + seg >= lookahead) {
                float ratio = (lookahead - acc) / seg;
                return {px + ratio * dx, py + ratio * dy};
            }
            acc += seg;
            px = nx;
            py = ny;
        }
        // Edge case: wrap around — just return opposite side of track
        int fallback = (start_idx + N / 2) % N;
        return {waypoints_[fallback].x, waypoints_[fallback].y};
    }

    // ── transform world point → vehicle frame (inline, no TF) ───
    //   cos_yaw / sin_yaw already computed in pose_callback.
    static std::array<float, 2>
    world_to_base(float gx, float gy,
                  float cx, float cy,
                  float cos_yaw, float sin_yaw) {
        float dx = gx - cx;
        float dy = gy - cy;
        return { dx * cos_yaw + dy * sin_yaw,
                -dx * sin_yaw + dy * cos_yaw };
    }

    // ── main callback ────────────────────────────────────────────
    void pose_callback(nav_msgs::msg::Odometry::ConstSharedPtr msg) {

        // 1. Unpack pose
        const auto& q = msg->pose.pose.orientation;
        // Fast yaw from quaternion (no euler conversion overhead)
        float yaw     = std::atan2(2.0f * (q.w * q.z + q.x * q.y),
                                   1.0f - 2.0f * (q.y * q.y + q.z * q.z));
        float cos_yaw = std::cos(yaw);
        float sin_yaw = std::sin(yaw);
        float cx      = static_cast<float>(msg->pose.pose.position.x);
        float cy      = static_cast<float>(msg->pose.pose.position.y);

        // 2. Nearest forward waypoint (KD-tree)
        int start_idx = nearest_forward(cx, cy, cos_yaw, sin_yaw);

        // 3. Lookahead distance from pre-computed table
        float lookahead = waypoints_[start_idx].l;
        float velocity  = waypoints_[start_idx].v;

        // 4. Goal point in world frame
        auto goal_world = compute_lookahead_pt(start_idx, lookahead);

        // 5. Transform to vehicle frame (no ROS TF lookup — saves ~0.5 ms)
        auto goal_local = world_to_base(goal_world[0], goal_world[1],
                                        cx, cy, cos_yaw, sin_yaw);

        // 6. Pure pursuit curvature + steering
        float lat_err       = goal_local[1];
        float curvature     = 2.0f * lat_err / (lookahead * lookahead);
        float steering_angle = std::atan(static_cast<float>(WHEELBASE) * curvature);
        // steering_angle = std::clamp(steering_angle,
        //                             -static_cast<float>(MAX_STEER_RAD),
        //                              static_cast<float>(MAX_STEER_RAD));

        // 7. Publish drive command
        ackermann_msgs::msg::AckermannDriveStamped drive_msg;
        drive_msg.header.stamp        = this->now();
        drive_msg.drive.steering_angle = static_cast<double>(steering_angle);
        drive_msg.drive.speed          = static_cast<double>(velocity);
        drive_pub_->publish(drive_msg);

        // 8. Goal marker (cheap — just one sphere)
        publish_goal_marker(goal_world[0], goal_world[1]);
    }

    // ── visualisation (2 Hz, not on hot path) ───────────────────
    void publish_waypoints() {
        visualization_msgs::msg::MarkerArray arr;
        arr.markers.reserve(waypoints_.size());
        const float range = static_cast<float>(V_MAX - V_MIN) + 1e-6f;
        auto now = this->now();

        for (std::size_t i = 0; i < waypoints_.size(); ++i) {
            visualization_msgs::msg::Marker m;
            m.header.frame_id = "map";
            m.header.stamp    = now;
            m.type   = visualization_msgs::msg::Marker::SPHERE;
            m.action = visualization_msgs::msg::Marker::ADD;
            m.id     = static_cast<int>(i);
            m.pose.position.x = waypoints_[i].x;
            m.pose.position.y = waypoints_[i].y;
            m.pose.position.z = 0.1;
            m.scale.x = m.scale.y = m.scale.z = 0.2;
            float v_norm = (waypoints_[i].v - static_cast<float>(V_MIN)) / range;
            m.color.r = v_norm;
            m.color.g = 1.0f - v_norm;
            m.color.b = 0.0f;
            m.color.a = 1.0f;
            arr.markers.push_back(m);
        }
        marker_pub_->publish(arr);
    }

    void publish_goal_marker(float gx, float gy) {
        visualization_msgs::msg::Marker m;
        m.header.frame_id = "map";
        m.header.stamp    = this->now();
        m.ns   = "goal";
        m.id   = 0;
        m.type   = visualization_msgs::msg::Marker::SPHERE;
        m.action = visualization_msgs::msg::Marker::ADD;
        m.pose.position.x = gx;
        m.pose.position.y = gy;
        m.pose.position.z = 0.2;
        m.scale.x = m.scale.y = m.scale.z = 0.35;
        m.color.r = 0.0f; m.color.g = 1.0f;
        m.color.b = 0.0f; m.color.a = 1.0f;

        visualization_msgs::msg::MarkerArray arr;
        arr.markers.push_back(m);
        goal_pub_->publish(arr);
    }
};

// ─────────────────────────────────────────────
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PurePursuit>());
    rclcpp::shutdown();
    return 0;
}