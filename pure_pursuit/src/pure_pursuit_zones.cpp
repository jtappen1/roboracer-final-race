#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <std_msgs/msg/bool.hpp>
#include <atomic>
#include <nanoflann.hpp>

#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <string>
#include <mutex>

// ─────────────────────────────────────────────
//  Compile-time constants (tune these)
// ─────────────────────────────────────────────
static constexpr double WHEELBASE     = 0.3;
static constexpr int    KD_CANDIDATES = 5;

// ─────────────────────────────────────────────
//  Lane enum
// ─────────────────────────────────────────────
enum class Lane { CENTER, LEFT, RIGHT };

// ─────────────────────────────────────────────
//  Zone definitions  ← TUNE THESE
//  start_idx / end_idx refer to indices in the
//  CENTER waypoint CSV (the reference path).
//  Zones define which side to overtake on when
//  CENTER is blocked. Checked in order; first match wins.
// ─────────────────────────────────────────────
struct Zone {
    int  start_idx;
    int  end_idx;
    Lane overtake_lane;
};

static const Zone ZONES[] = {
    // { start, end, Lane::LEFT or Lane::RIGHT }
    { 0, 193, Lane::RIGHT },

    // Add more zones here as needed
};
static constexpr int NUM_ZONES = static_cast<int>(sizeof(ZONES) / sizeof(ZONES[0]));

// ─────────────────────────────────────────────
//  Data Structures
// ─────────────────────────────────────────────
struct Waypoint {
    float x, y, v, l;
};

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

// ─────────────────────────────────────────────
//  Node Definition
// ─────────────────────────────────────────────
class PurePursuitLaneSwitcher : public rclcpp::Node {
public:
    PurePursuitLaneSwitcher() : Node("pp_lane_switcher_node") {
        // ── Parameters ──────────────────────────────────────
        this->declare_parameter<std::string>("center_csv", "/home/nvidia/ros2_ws/src/roboracer-final-race/pure_pursuit/path/waypoints_oncar.csv");
        this->declare_parameter<std::string>("left_csv",   "/home/nvidia/ros2_ws/src/roboracer-final-race/pure_pursuit/path/waypoints_denser5_slower.csv");
        this->declare_parameter<std::string>("right_csv",  "/home/nvidia/ros2_ws/src/roboracer-final-race/pure_pursuit/path/waypoints_edited.csv");
        this->declare_parameter<float>("blocking_radius",       0.5f);
        this->declare_parameter<float>("lookahead_window_dist", 5.0f);
        this->declare_parameter<int>("clear_confirm_count",     30);
        this->declare_parameter<bool>("enable_visualization",   false);

        const auto center_path = this->get_parameter("center_csv").as_string();
        const auto left_path   = this->get_parameter("left_csv").as_string();
        const auto right_path  = this->get_parameter("right_csv").as_string();
        blocking_radius_       = this->get_parameter("blocking_radius").as_double();
        lookahead_window_dist_ = this->get_parameter("lookahead_window_dist").as_double();
        clear_confirm_count_   = this->get_parameter("clear_confirm_count").as_int();
        enable_visualization_  = this->get_parameter("enable_visualization").as_bool();

        // ── Load waypoints & build KD-trees ────────────────
        load_waypoints(center_path, center_waypoints_);
        center_cloud_ = std::make_unique<WaypointCloud>(center_waypoints_);
        center_tree_  = std::make_unique<KDTree>(2, *center_cloud_, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        center_tree_->buildIndex();

        load_waypoints(left_path, left_waypoints_);
        left_cloud_ = std::make_unique<WaypointCloud>(left_waypoints_);
        left_tree_  = std::make_unique<KDTree>(2, *left_cloud_, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        left_tree_->buildIndex();

        load_waypoints(right_path, right_waypoints_);
        right_cloud_ = std::make_unique<WaypointCloud>(right_waypoints_);
        right_tree_  = std::make_unique<KDTree>(2, *right_cloud_, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        right_tree_->buildIndex();

        // ── ROS interfaces ─────────────────────────────────
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/pf/pose/odom", rclcpp::SensorDataQoS(),
            [this](nav_msgs::msg::Odometry::ConstSharedPtr msg){ pose_callback(msg); });

        obstacles_sub_ = this->create_subscription<geometry_msgs::msg::PoseArray>(
            "/obstacles/centroids", 10,
            [this](geometry_msgs::msg::PoseArray::ConstSharedPtr msg){ obstacles_callback(msg); });
        
        brake_sub_ = this->create_subscription<std_msgs::msg::Bool>(
    "/aeb_brake", 10,
    [this](std_msgs::msg::Bool::ConstSharedPtr msg){ brake_active_ = msg->data; });

        drive_pub_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
            "/drive", rclcpp::SystemDefaultsQoS());

        if (enable_visualization_) {
            marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
                "/pure_pursuit/lane_markers", rclcpp::ServicesQoS());
            goal_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
                "/pure_pursuit/goal_marker", rclcpp::ServicesQoS());
            vis_timer_ = this->create_wall_timer(
                std::chrono::milliseconds(500), [this](){ publish_markers(); });
        }

        RCLCPP_INFO(get_logger(),
            "Initialized. Center: %zu pts | Left: %zu pts | Right: %zu pts | Vis: %s",
            center_waypoints_.size(), left_waypoints_.size(), right_waypoints_.size(),
            enable_visualization_ ? "ON" : "OFF");
    }

private:
    std::vector<Waypoint> center_waypoints_;
    std::vector<Waypoint> left_waypoints_;
    std::vector<Waypoint> right_waypoints_;

    std::unique_ptr<WaypointCloud> center_cloud_, left_cloud_, right_cloud_;
    std::unique_ptr<KDTree>        center_tree_,  left_tree_,  right_tree_;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr       odom_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr obstacles_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr brake_sub_;
    std::atomic<bool> brake_active_{false};
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr       marker_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr       goal_pub_;
    rclcpp::TimerBase::SharedPtr vis_timer_;

    // Config
    float blocking_radius_;
    float lookahead_window_dist_;
    int   clear_confirm_count_;
    bool  enable_visualization_ = false;

    // State
    Lane active_lane_          = Lane::CENTER;
    int  obstacle_clear_count_ = 0;
    int  last_center_idx_      = 0;
    int  last_left_idx_        = 0;
    int  last_right_idx_       = 0;

    std::vector<std::array<float, 2>> obstacles_;
    std::mutex obs_mutex_;

    // ── File Loading ────────────────────────────────────────────────────────
    void load_waypoints(const std::string& path, std::vector<Waypoint>& wp_vec) {
        std::ifstream f(path);
        if (!f.is_open()) throw std::runtime_error("Cannot open waypoints: " + path);
        wp_vec.reserve(4096);
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            std::string tok;
            Waypoint wp{};
            if (std::getline(ss, tok, ',')) wp.x = std::stof(tok);
            if (std::getline(ss, tok, ',')) wp.y = std::stof(tok);
            if (std::getline(ss, tok, ',')) wp.v = std::stof(tok);
            if (std::getline(ss, tok, ',')) wp.l = std::stof(tok);
            wp_vec.push_back(wp);
        }
    }

    // ── Obstacle Callback ───────────────────────────────────────────────────
    void obstacles_callback(geometry_msgs::msg::PoseArray::ConstSharedPtr msg) {
        std::lock_guard<std::mutex> lock(obs_mutex_);
        obstacles_.clear();
        for (const auto& pose : msg->poses)
            obstacles_.push_back({static_cast<float>(pose.position.x),
                                  static_cast<float>(pose.position.y)});
    }

    // ── Zone Lookup — returns which lane to overtake on at this center index ─
    Lane zone_for_idx(int center_idx) const {
        for (int z = 0; z < NUM_ZONES; ++z)
            if (center_idx >= ZONES[z].start_idx && center_idx <= ZONES[z].end_idx)
                return ZONES[z].overtake_lane;
        return Lane::LEFT; // default overtake side if no zone defined
    }

    // ── Lane Blockage Evaluator ─────────────────────────────────────────────
    bool is_lane_blocked(const std::vector<Waypoint>& lane, int start_idx) {
        std::lock_guard<std::mutex> lock(obs_mutex_);
        if (obstacles_.empty()) return false;

        const int N = static_cast<int>(lane.size());
        float acc_dist = 0.0f;
        float px = lane[start_idx].x;
        float py = lane[start_idx].y;

        for (int i = 0; i < N; ++i) {
            int idx = (start_idx + i) % N;
            float nx = lane[idx].x;
            float ny = lane[idx].y;
            acc_dist += std::hypot(nx - px, ny - py);
            if (acc_dist > lookahead_window_dist_) break;

            for (const auto& obs : obstacles_)
                if (std::hypot(nx - obs[0], ny - obs[1]) < blocking_radius_)
                    return true;

            px = nx; py = ny;
        }
        return false;
    }

    // ── Fast nearest-waypoint (ahead of car) via KD-tree ────────────────────
    int nearest_forward(float cx, float cy, float cos_yaw, float sin_yaw,
                        const std::vector<Waypoint>& wps, KDTree* tree, int& last_idx) {
        float query[2] = {cx, cy};
        std::array<std::size_t, KD_CANDIDATES> indices;
        std::array<float,       KD_CANDIDATES> dists_sq;

        nanoflann::KNNResultSet<float> rs(KD_CANDIDATES);
        rs.init(indices.data(), dists_sq.data());
        tree->findNeighbors(rs, query, nanoflann::SearchParameters());

        int   best_idx  = -1;
        float best_dist = std::numeric_limits<float>::max();
        for (int k = 0; k < KD_CANDIDATES; ++k) {
            int i = static_cast<int>(indices[k]);
            float dx = wps[i].x - cx;
            float dy = wps[i].y - cy;
            if ((dx * cos_yaw + dy * sin_yaw) > 0.0f && dists_sq[k] < best_dist) {
                best_dist = dists_sq[k];
                best_idx  = i;
            }
        }
        if (best_idx < 0) best_idx = static_cast<int>(indices[0]);
        last_idx = best_idx;
        return best_idx;
    }

    // ── Lookahead target finder ─────────────────────────────────────────────
    std::array<float, 2> compute_lookahead_pt(const std::vector<Waypoint>& wps,
                                              int start_idx, float lookahead) const {
        const int N = static_cast<int>(wps.size());
        float acc = 0.0f;
        float px = wps[start_idx].x;
        float py = wps[start_idx].y;

        for (int i = 1; i <= N; ++i) {
            int idx = (start_idx + i) % N;
            float nx = wps[idx].x, ny = wps[idx].y;
            float seg = std::hypot(nx - px, ny - py);
            if (acc + seg >= lookahead) {
                float ratio = (lookahead - acc) / seg;
                return {px + ratio * (nx - px), py + ratio * (ny - py)};
            }
            acc += seg; px = nx; py = ny;
        }
        int fallback = (start_idx + N / 2) % N;
        return {wps[fallback].x, wps[fallback].y};
    }

    // ── World-to-Vehicle frame ──────────────────────────────────────────────
    static std::array<float, 2> world_to_base(float gx, float gy, float cx, float cy,
                                              float cos_yaw, float sin_yaw) {
        float dx = gx - cx, dy = gy - cy;
        return { dx * cos_yaw + dy * sin_yaw,
                -dx * sin_yaw + dy * cos_yaw };
    }

    // ── Main Control Loop ───────────────────────────────────────────────────
    void pose_callback(nav_msgs::msg::Odometry::ConstSharedPtr msg) {
        const auto& q = msg->pose.pose.orientation;
        float yaw = std::atan2(2.0f * (q.w * q.z + q.x * q.y),
                               1.0f - 2.0f * (q.y * q.y + q.z * q.z));
        float cos_yaw = std::cos(yaw);
        float sin_yaw = std::sin(yaw);
        float cx = static_cast<float>(msg->pose.pose.position.x);
        float cy = static_cast<float>(msg->pose.pose.position.y);

        // 1. Localize on all three paths
        int center_idx = nearest_forward(cx, cy, cos_yaw, sin_yaw, center_waypoints_, center_tree_.get(), last_center_idx_);
        int left_idx   = nearest_forward(cx, cy, cos_yaw, sin_yaw, left_waypoints_,   left_tree_.get(),   last_left_idx_);
        int right_idx  = nearest_forward(cx, cy, cos_yaw, sin_yaw, right_waypoints_,  right_tree_.get(),  last_right_idx_);

        // 2. Always check if CENTER is blocked
        bool center_blocked = is_lane_blocked(center_waypoints_, center_idx);

        // 3. State machine
        //    Default: CENTER always.
        //    Only switch to zone-prescribed lane when CENTER is blocked.
        //    Return to CENTER once clear (with hysteresis).
        if (active_lane_ == Lane::CENTER) {
            if (center_blocked) {
                // Use the zone table to decide which side to overtake on
                Lane overtake = zone_for_idx(center_idx);
                RCLCPP_INFO(get_logger(), "[OBSTACLE] CENTER blocked -> overtaking via %s",
                    lane_name(overtake));
                active_lane_ = overtake;
                obstacle_clear_count_ = 0;
            }
        } else {
            // We are on an overtake lane — wait for CENTER to clear before returning
            if (!center_blocked) {
                obstacle_clear_count_++;
                if (obstacle_clear_count_ >= clear_confirm_count_) {
                    RCLCPP_INFO(get_logger(), "[CLEAR] Returning to CENTER");
                    active_lane_ = Lane::CENTER;
                    obstacle_clear_count_ = 0;
                }
            } else {
                obstacle_clear_count_ = 0; // still blocked, reset hysteresis counter
            }
        }

        // 4. Control targets from active lane
        const auto& target_wps = lane_waypoints(active_lane_);
        int target_idx = lane_idx(active_lane_, center_idx, left_idx, right_idx);

        float lookahead      = target_wps[target_idx].l;
        float velocity       = target_wps[target_idx].v;

        // 5. Lookahead and steering
        auto goal_world      = compute_lookahead_pt(target_wps, target_idx, lookahead);
        auto goal_local      = world_to_base(goal_world[0], goal_world[1], cx, cy, cos_yaw, sin_yaw);
        float lat_err        = goal_local[1];
        float curvature      = 2.0f * lat_err / (lookahead * lookahead);
        float steering_angle = std::atan(static_cast<float>(WHEELBASE) * curvature);

        // 6. Publish
        ackermann_msgs::msg::AckermannDriveStamped drive_msg;
        drive_msg.header.stamp         = this->now();
        drive_msg.drive.steering_angle = static_cast<double>(steering_angle);
        drive_msg.drive.speed          = brake_active_.load() ? 0.0 : static_cast<double>(velocity);
        // ===== Debug Print =====
        if (brake_active_.load()) {
            RCLCPP_WARN(this->get_logger(),
                "[AEB STOP] Brake Active | speed=0 | lane=%s | center_idx=%d | steering=%.3f",
                lane_name(active_lane_),
                center_idx,
                steering_angle);
        } else {
            RCLCPP_INFO(this->get_logger(),
                "[DRIVE] speed=%.2f | steering=%.3f | lane=%s",
                velocity,
                steering_angle,
                lane_name(active_lane_));
        }
        drive_pub_->publish(drive_msg);


        if (enable_visualization_)
            publish_goal_marker(goal_world[0], goal_world[1]);
    }

    // ── Lane helpers ────────────────────────────────────────────────────────
    const std::vector<Waypoint>& lane_waypoints(Lane l) const {
        if (l == Lane::LEFT)  return left_waypoints_;
        if (l == Lane::RIGHT) return right_waypoints_;
        return center_waypoints_;
    }

    static int lane_idx(Lane l, int ci, int li, int ri) {
        if (l == Lane::LEFT)  return li;
        if (l == Lane::RIGHT) return ri;
        return ci;
    }

    static const char* lane_name(Lane l) {
        switch (l) {
            case Lane::CENTER: return "CENTER";
            case Lane::LEFT:   return "LEFT";
            case Lane::RIGHT:  return "RIGHT";
        }
        return "?";
    }

    // ── Visualization ───────────────────────────────────────────────────────
    void publish_markers() {
        visualization_msgs::msg::MarkerArray arr;
        auto now = this->now();

        auto add_lane_markers = [&](const std::vector<Waypoint>& wps, Lane lane_type) {
            bool is_active = (lane_type == active_lane_);
            for (std::size_t i = 0; i < wps.size(); i += 2) {
                visualization_msgs::msg::Marker m;
                m.header.frame_id = "map";
                m.header.stamp    = now;
                m.ns     = lane_name(lane_type);
                m.type   = visualization_msgs::msg::Marker::SPHERE;
                m.action = visualization_msgs::msg::Marker::ADD;
                m.id     = static_cast<int>(i);
                m.pose.position.x = wps[i].x;
                m.pose.position.y = wps[i].y;
                m.pose.position.z = 0.05;
                m.scale.x = m.scale.y = m.scale.z = is_active ? 0.2f : 0.1f;

                // CENTER=red, LEFT=green, RIGHT=blue. Dim when inactive.
                m.color.r = 0.0f; m.color.g = 0.0f; m.color.b = 0.0f; m.color.a = 1.0f;
                if (lane_type == Lane::CENTER) m.color.r = is_active ? 1.0f : 0.4f;
                if (lane_type == Lane::LEFT)   m.color.g = is_active ? 1.0f : 0.4f;
                if (lane_type == Lane::RIGHT)  m.color.b = is_active ? 1.0f : 0.4f;

                arr.markers.push_back(m);
            }
        };

        add_lane_markers(center_waypoints_, Lane::CENTER);
        add_lane_markers(left_waypoints_,   Lane::LEFT);
        add_lane_markers(right_waypoints_,  Lane::RIGHT);
        marker_pub_->publish(arr);
    }

    void publish_goal_marker(float gx, float gy) {
        visualization_msgs::msg::Marker m;
        m.header.frame_id = "map";
        m.header.stamp    = this->now();
        m.ns     = "pure_pursuit_goal";
        m.id     = 0;
        m.type   = visualization_msgs::msg::Marker::SPHERE;
        m.action = visualization_msgs::msg::Marker::ADD;
        m.pose.position.x = gx;
        m.pose.position.y = gy;
        m.pose.position.z = 0.2;
        m.scale.x = m.scale.y = m.scale.z = 0.4f;
        m.color.r = 1.0f; m.color.g = 1.0f; m.color.b = 0.0f; m.color.a = 1.0f;

        visualization_msgs::msg::MarkerArray arr;
        arr.markers.push_back(m);
        goal_pub_->publish(arr);
    }
};

// ─────────────────────────────────────────────
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PurePursuitLaneSwitcher>());
    rclcpp::shutdown();
    return 0;
}
