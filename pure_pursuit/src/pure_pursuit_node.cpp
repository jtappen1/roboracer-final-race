#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <nanoflann.hpp>

#include <Eigen/Core>

#include "ament_index_cpp/get_package_share_directory.hpp"

#include "overtake_msgs/msg/tracked_obstacle_array.hpp"

#include "final_race_pure_pursuit/frenet_overtake_planner.hpp"
#include "final_race_pure_pursuit/frenet_utils.hpp"
#include "final_race_pure_pursuit/mpc_utils.hpp"

#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <string>
#include <memory>

// ─────────────────────────────────────────────
//  Compile-time constants (tune these)
// ─────────────────────────────────────────────
static constexpr double WHEELBASE      = 0.33;
static constexpr double L_MIN          = 0.5;
static constexpr double L_MAX          = 3.0;
static constexpr double V_MIN          = 2.0;
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
    PurePursuit() : Node("pure_pursuit_node") {

        // ── load waypoints ──────────────────────────────────────
        this->declare_parameter<std::string>("waypoints_path", "path/waypoints_optimized_traj.csv");
        const auto wp_path = resolve_waypoints_path(
            this->get_parameter("waypoints_path").as_string());
        load_waypoints(wp_path);

        // ── build KD-tree once ──────────────────────────────────
        cloud_  = std::make_unique<WaypointCloud>(waypoints_);
        kdtree_ = std::make_unique<KDTree>(2, *cloud_,
                      nanoflann::KDTreeSingleIndexAdaptorParams(10 /*leaf_max_size*/));
        kdtree_->buildIndex();

        // ── Frenet overtake planner parameters ──────────────────
        this->declare_parameter<bool>("enable_overtaking", true);
        this->declare_parameter<std::string>("obstacle_topic", "/overtake/tracked_obstacles");
        this->declare_parameter<int>("horizon_steps", 10);
        this->declare_parameter<double>("planning_dt", 0.1);
        this->declare_parameter<double>("max_speed", 5.0);
        this->declare_parameter<double>("min_speed", 3.0);
        this->declare_parameter<double>("track_half_width", 0.85);

        enable_overtaking_   = this->get_parameter("enable_overtaking").as_bool();
        horizon_steps_       = static_cast<int>(this->get_parameter("horizon_steps").as_int());
        planning_dt_         = this->get_parameter("planning_dt").as_double();
        const double planner_max_speed   = this->get_parameter("max_speed").as_double();
        const double planner_min_speed   = this->get_parameter("min_speed").as_double();
        const double planner_track_half_width = this->get_parameter("track_half_width").as_double();

        if (enable_overtaking_) {
            build_planner(planner_max_speed, planner_min_speed, planner_track_half_width);

            const auto obstacle_topic = this->get_parameter("obstacle_topic").as_string();
            // Use the default reliable QoS — the slow obstacle publisher
            // (and any real obstacle tracker) is reliable, so a SensorData
            // (best-effort) sub here would silently never receive anything.
            obstacle_sub_ = this->create_subscription<overtake_msgs::msg::TrackedObstacleArray>(
                obstacle_topic, rclcpp::QoS(rclcpp::KeepLast(10)),
                [this](overtake_msgs::msg::TrackedObstacleArray::ConstSharedPtr msg){
                    obstacle_callback(msg);
                });

            local_path_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
                "/pure_pursuit/local_path", rclcpp::ServicesQoS());

            RCLCPP_INFO(get_logger(),
                        "Frenet overtake planner enabled — listening on %s",
                        obstacle_topic.c_str());
        }

        // ── ROS interfaces ──────────────────────────────────────
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "ego_racecar/odom", rclcpp::SensorDataQoS(),
            [this](nav_msgs::msg::Odometry::ConstSharedPtr msg){ pose_callback(msg); });

        drive_pub_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
            "/drive", rclcpp::SystemDefaultsQoS());

        // Visualisation (low-priority, 2 Hz)
        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/pure_pursuit/waypoint_markers", rclcpp::ServicesQoS());
        vis_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(500), [this](){ publish_waypoints(); });

        goal_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/pure_pursuit/goal_marker", rclcpp::ServicesQoS());

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

    // Frenet overtake bridge
    bool   enable_overtaking_ = false;
    int    horizon_steps_     = 10;
    double planning_dt_       = 0.1;
    std::unique_ptr<mpc::FrenetOvertakePlanner> planner_;
    rclcpp::Subscription<overtake_msgs::msg::TrackedObstacleArray>::SharedPtr obstacle_sub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr local_path_pub_;
    std::string last_mode_ = "nominal";

    // last-known closest index (warm-start — avoids cold search every frame)
    int last_idx_ = 0;

    // ── resolve a (possibly relative) waypoints path ────────────
    //   Mirrors the python launcher: a leading '/' is treated as an
    //   absolute path; otherwise we prepend the installed package
    //   share directory so the same YAML works on car and in sim.
    std::string resolve_waypoints_path(const std::string& path) const {
        if (path.empty() || path.front() == '/') {
            return path;
        }
        try {
            const std::string share_dir =
                ament_index_cpp::get_package_share_directory(
                    "final_race_pure_pursuit");
            return share_dir + "/" + path;
        } catch (const std::exception& ex) {
            RCLCPP_WARN(get_logger(),
                        "Could not resolve package share dir (%s) — using path as-is",
                        ex.what());
            return path;
        }
    }

    // ── waypoint loading ────────────────────────────────────────
    //   Auto-detects the column layout from the first non-comment line:
    //     • 7 cols, ';' delimited → optimized trajectory format
    //         (s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2)
    //         — x/y come from cols 1/2 and velocity from col 5; the
    //           per-waypoint lookahead is synthesised from the velocity
    //           profile since the trajectory file does not carry one.
    //     • 4 cols, ',' delimited → legacy (x, y, lookahead, velocity)
    //     • 3 cols, ',' delimited → older (x, y, velocity); lookahead
    //                                synthesised from velocity.
    //   The lookahead synthesis mirrors the python pursuit:
    //       v_norm = (v - V_MIN) / (V_MAX - V_MIN)
    //       l = clamp(v_norm * L_MAX, L_MIN, L_MAX)
    //   so corners (low v) get a short lookahead and straights get long.
    void load_waypoints(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open())
            throw std::runtime_error("Cannot open waypoints: " + path);

        waypoints_.reserve(4096);
        std::string line;
        char delim = ',';
        int  cols  = 0;

        auto split = [](const std::string& s, char d) {
            std::vector<std::string> out;
            std::stringstream ss(s);
            std::string tok;
            while (std::getline(ss, tok, d)) out.push_back(tok);
            return out;
        };

        auto synth_lookahead = [](float v) {
            const float v_norm = (v - static_cast<float>(V_MIN)) /
                                 static_cast<float>(V_MAX - V_MIN + 1e-6);
            const float raw    = v_norm * static_cast<float>(L_MAX);
            return std::max(static_cast<float>(L_MIN),
                            std::min(static_cast<float>(L_MAX), raw));
        };

        while (std::getline(f, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty() || line[0] == '#') continue;  // skip comments and blanks

            // First data row: pick the delimiter & remember column count.
            if (cols == 0) {
                delim = (line.find(';') != std::string::npos) ? ';' : ',';
                cols  = static_cast<int>(split(line, delim).size());
                RCLCPP_INFO(get_logger(),
                            "Waypoint format: %d cols, delim='%c'",
                            cols, delim);
            }

            const auto fields = split(line, delim);
            Waypoint wp{};
            if (cols >= 7 && delim == ';' && fields.size() >= 6) {
                // Optimized trajectory: s; x; y; psi; kappa; vx; ax
                wp.x = std::stof(fields[1]);
                wp.y = std::stof(fields[2]);
                wp.v = std::stof(fields[5]);
                wp.l = synth_lookahead(wp.v);
            } else if (cols >= 4 && fields.size() >= 4) {
                // Legacy: x, y, lookahead, velocity
                wp.x = std::stof(fields[0]);
                wp.y = std::stof(fields[1]);
                wp.l = std::stof(fields[2]);
                wp.v = std::stof(fields[3]);
            } else if (cols >= 3 && fields.size() >= 3) {
                // Older: x, y, velocity (no lookahead column)
                wp.x = std::stof(fields[0]);
                wp.y = std::stof(fields[1]);
                wp.v = std::stof(fields[2]);
                wp.l = synth_lookahead(wp.v);
            } else {
                continue;  // skip malformed rows
            }
            waypoints_.push_back(wp);
        }
        RCLCPP_INFO(get_logger(), "Loaded %zu waypoints from %s",
                    waypoints_.size(), path.c_str());
    }

    // ── instantiate FrenetOvertakePlanner from raceline waypoints ──
    //   Re-uses the same loaded raceline as both the nominal pursuit
    //   target and the planner's centerline reference.
    void build_planner(double max_speed, double min_speed, double track_half_width) {
        const Eigen::Index N = static_cast<Eigen::Index>(waypoints_.size());
        if (N < 3) {
            RCLCPP_WARN(get_logger(),
                        "Not enough waypoints to build Frenet planner — overtaking disabled");
            enable_overtaking_ = false;
            return;
        }

        Eigen::VectorXd ref_x(N);
        Eigen::VectorXd ref_y(N);
        Eigen::VectorXd ref_v(N);
        Eigen::VectorXd ref_yaw(N);
        for (Eigen::Index i = 0; i < N; ++i) {
            ref_x(i) = static_cast<double>(waypoints_[static_cast<std::size_t>(i)].x);
            ref_y(i) = static_cast<double>(waypoints_[static_cast<std::size_t>(i)].y);
            ref_v(i) = static_cast<double>(waypoints_[static_cast<std::size_t>(i)].v);
        }
        for (Eigen::Index i = 0; i < N; ++i) {
            const Eigen::Index j = (i + 1) % N;
            ref_yaw(i) = std::atan2(ref_y(j) - ref_y(i), ref_x(j) - ref_x(i));
        }

        planner_ = std::make_unique<mpc::FrenetOvertakePlanner>(
            ref_x, ref_y, ref_yaw, ref_v, planning_dt_, horizon_steps_,
            max_speed, min_speed, track_half_width);
    }

    // ── obstacle ingest — push tracks into planner ──────────────
    void obstacle_callback(overtake_msgs::msg::TrackedObstacleArray::ConstSharedPtr msg) {
        if (!planner_) return;
        const double stamp_sec = static_cast<double>(msg->header.stamp.sec) +
                                 static_cast<double>(msg->header.stamp.nanosec) * 1e-9;
        planner_->updateObstacles(*msg, stamp_sec);
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
        kdtree_->findNeighbors(rs, query, nanoflann::SearchParameters());

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

    // ── build a short-horizon nominal ref-path along the raceline ─
    //   Used as the planner's "if everything is clear" baseline. We
    //   walk arc-length forward from the ego's nearest forward waypoint,
    //   sampling once per planning_dt_ * current_speed metres so the
    //   spacing roughly matches the planner's internal time grid.
    Eigen::MatrixXd build_nominal_ref_path(int start_idx, double current_speed) const {
        const int short_count = horizon_steps_ + 1;
        Eigen::MatrixXd path(4, short_count);
        const int N = static_cast<int>(waypoints_.size());

        // Arc-length step ≈ v * dt, with a sane lower bound so the
        // path still spans space when ego is stationary.
        const double step_len =
            std::max(std::abs(current_speed) * planning_dt_, 0.10);

        // Seed first point at the start waypoint
        double px = waypoints_[start_idx].x;
        double py = waypoints_[start_idx].y;
        path(0, 0) = px;
        path(1, 0) = py;
        path(2, 0) = waypoints_[start_idx].v;

        double acc       = 0.0;
        int    next_step = 1;
        // Walk forward up to four laps just in case step_len is huge
        for (int i = 1; i <= 4 * N && next_step < short_count; ++i) {
            const int nidx = (start_idx + i) % N;
            const double nx = waypoints_[nidx].x;
            const double ny = waypoints_[nidx].y;
            const double dx = nx - px;
            const double dy = ny - py;
            const double seg = std::sqrt(dx * dx + dy * dy);

            while (next_step < short_count &&
                   acc + seg >= next_step * step_len) {
                const double rem  = next_step * step_len - acc;
                const double ratio = (seg > 1e-6) ? rem / seg : 0.0;
                path(0, next_step) = px + ratio * dx;
                path(1, next_step) = py + ratio * dy;
                path(2, next_step) = waypoints_[nidx].v;
                ++next_step;
            }
            acc += seg;
            px = nx;
            py = ny;
        }
        // Pad with the last sample if we somehow ran short.
        for (; next_step < short_count; ++next_step) {
            path.col(next_step) = path.col(next_step - 1);
        }
        // Yaw row from finite differences along x/y rows.
        for (int k = 0; k < short_count - 1; ++k) {
            path(3, k) = std::atan2(path(1, k + 1) - path(1, k),
                                    path(0, k + 1) - path(0, k));
        }
        path(3, short_count - 1) =
            short_count > 1 ? path(3, short_count - 2) : 0.0;
        return path;
    }

    // ── walk arc-length along a 4×N planner ref_path to the lookahead ──
    //   The path may not include the ego pose, so we anchor the walk at
    //   (cx, cy) and include the gap to the first ref point in the arc
    //   distance — this keeps the lookahead consistent with the global
    //   raceline version above.
    static std::array<float, 2>
    compute_local_lookahead_pt(const Eigen::MatrixXd& ref_path,
                               float cx, float cy, float lookahead) {
        const int M = static_cast<int>(ref_path.cols());
        if (M == 0) return {cx, cy};

        // Find the path point closest to ego — use it as the start of
        // the forward walk so we don't accumulate distance on segments
        // we have already overshot.
        int start = 0;
        double min_d2 = std::numeric_limits<double>::infinity();
        for (int i = 0; i < M; ++i) {
            const double dx = ref_path(0, i) - cx;
            const double dy = ref_path(1, i) - cy;
            const double d2 = dx * dx + dy * dy;
            if (d2 < min_d2) { min_d2 = d2; start = i; }
        }

        double acc = 0.0;
        double px  = cx;
        double py  = cy;
        for (int i = start; i < M; ++i) {
            const double nx  = ref_path(0, i);
            const double ny  = ref_path(1, i);
            const double dx  = nx - px;
            const double dy  = ny - py;
            const double seg = std::sqrt(dx * dx + dy * dy);
            if (acc + seg >= lookahead) {
                const double rem  = lookahead - acc;
                const double ratio = (seg > 1e-6) ? rem / seg : 0.0;
                return {static_cast<float>(px + ratio * dx),
                        static_cast<float>(py + ratio * dy)};
            }
            acc += seg;
            px = nx;
            py = ny;
        }
        // Fallback: end of local plan if it is shorter than lookahead.
        return {static_cast<float>(ref_path(0, M - 1)),
                static_cast<float>(ref_path(1, M - 1))};
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

        // 4. Goal point in world frame — defaults to the nominal raceline,
        //    then overridden by the Frenet overtake planner only when an
        //    obstacle is actively shaping the trajectory (overtake / follow).
        //    Rejoin/nominal stay on the original raceline pursuit so we keep
        //    the per-waypoint lookahead/velocity tuned for tight corners and
        //    don't drive the truncated, speed-clamped local path when there
        //    is no real reason to deviate from the raceline.
        std::array<float, 2> goal_world = compute_lookahead_pt(start_idx, lookahead);

        if (enable_overtaking_ && planner_) {
            // Ego speed: forward component of body-frame twist (gym/odom convention)
            const double ego_speed =
                std::abs(static_cast<double>(msg->twist.twist.linear.x));

            mpc::State state;
            state.x   = static_cast<double>(cx);
            state.y   = static_cast<double>(cy);
            state.yaw = static_cast<double>(yaw);
            state.v   = ego_speed;

            const Eigen::MatrixXd nominal_ref_path =
                build_nominal_ref_path(start_idx, ego_speed);

            const auto plan_result = planner_->plan(state, nominal_ref_path);
            const Eigen::MatrixXd& ref_path = plan_result.first;
            const mpc::PlanInfo&   info     = plan_result.second;

            const bool obstacle_active =
                (info.mode == "overtake" || info.mode == "follow");
            if (obstacle_active && ref_path.cols() > 0 && ref_path.rows() >= 3) {
                // Drive the planner's local path while it is actively reacting
                // to a tracked obstacle.
                goal_world =
                    compute_local_lookahead_pt(ref_path, cx, cy, lookahead);

                // Velocity from the local plan — pick the next step (col 1)
                // so we don't lock onto the ego's current speed at col 0.
                const Eigen::Index v_col =
                    std::min<Eigen::Index>(1, ref_path.cols() - 1);
                const double v_local = ref_path(2, v_col);
                if (std::isfinite(v_local) && v_local > 0.0) {
                    velocity = static_cast<float>(
                        std::min(static_cast<double>(V_MAX), v_local));
                }
            }

            if (info.mode != last_mode_) {
                RCLCPP_INFO(get_logger(),
                            "Frenet planner mode → %s (label=%s, side=%d, cost=%.3f)",
                            info.mode.c_str(), info.label.c_str(), info.side,
                            info.cost);
                last_mode_ = info.mode;
            }

            publish_local_path(ref_path, info.mode);
        }

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

    // ── visualise the planner's local trajectory ────────────────
    //   Colour depends on mode so it's obvious in foxglove/RViz which
    //   plan is currently driving the car.
    void publish_local_path(const Eigen::MatrixXd& ref_path,
                            const std::string&     mode) {
        if (!local_path_pub_) return;
        const Eigen::Index M = ref_path.cols();
        if (M <= 0 || ref_path.rows() < 2) return;

        visualization_msgs::msg::MarkerArray arr;
        visualization_msgs::msg::Marker line;
        line.header.frame_id = "map";
        line.header.stamp    = this->now();
        line.ns              = "frenet_plan";
        line.id              = 0;
        line.type            = visualization_msgs::msg::Marker::LINE_STRIP;
        line.action          = visualization_msgs::msg::Marker::ADD;
        line.scale.x         = 0.06;
        if (mode == "overtake") {
            line.color.r = 1.0f; line.color.g = 0.2f; line.color.b = 0.2f;
        } else if (mode == "follow") {
            line.color.r = 1.0f; line.color.g = 0.8f; line.color.b = 0.0f;
        } else if (mode == "rejoin") {
            line.color.r = 0.2f; line.color.g = 0.6f; line.color.b = 1.0f;
        } else {
            line.color.r = 0.6f; line.color.g = 0.6f; line.color.b = 0.6f;
        }
        line.color.a = 0.9f;
        line.points.reserve(static_cast<std::size_t>(M));
        for (Eigen::Index i = 0; i < M; ++i) {
            geometry_msgs::msg::Point p;
            p.x = ref_path(0, i);
            p.y = ref_path(1, i);
            p.z = 0.15;
            line.points.push_back(p);
        }
        arr.markers.push_back(line);
        local_path_pub_->publish(arr);
    }
};

// ─────────────────────────────────────────────
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PurePursuit>());
    rclcpp::shutdown();
    return 0;
}
