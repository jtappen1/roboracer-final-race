#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <deque>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "std_msgs/msg/color_rgba.hpp"
#include "std_msgs/msg/header.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2/exceptions.h"

using namespace std::chrono_literals;

// ─── Tracker state ─────────────────────────────────────────────────────────
struct ObstacleTrack
{
    int id;
    double x;          // running EMA of position in map frame
    double y;
    int hits;          // total times this track was associated
    int misses;        // consecutive frames without an update
    bool confirmed;    // promoted to published after enough hits
};

// ─── Map (occupancy grid loaded from .yaml + .pgm) ─────────────────────────
struct OccupancyMap
{
    bool loaded = false;
    int width = 0;
    int height = 0;
    double resolution = 0.0;     // meters per cell
    double origin_x = 0.0;       // world coords of cell (0,0)
    double origin_y = 0.0;
    int occ_thresh = 65;         // cells with value > this are "occupied" (after PGM normalization)
    std::vector<uint8_t> data;   // row-major, 0=free, 100=occupied (ROS convention-ish)
};


class LidarObstacleNode : public rclcpp::Node
{
public:
    LidarObstacleNode() : Node("lidar_obstacle_node"), next_track_id_(0)
    {
        // ── Parameters ──────────────────────────────────────────────────────
        this->declare_parameter("scan_topic", "/scan");
        this->declare_parameter("map_frame", "map");
        this->declare_parameter("obstacle_radius", 0.5);

        this->declare_parameter("min_range", 0.15);
        this->declare_parameter("max_range", 3.0);
        this->declare_parameter("fov_fraction", 0.1);

        this->declare_parameter("gap_threshold", 0.15);
        this->declare_parameter("min_segment_points", 10);
        this->declare_parameter("max_segment_points", 80);

        this->declare_parameter("max_extended_segment_points", 200);
        this->declare_parameter("max_distance", 3.0);

        this->declare_parameter("waypoints_csv", "/home/nvidia/ros2_ws/src/roboracer-final-race/pure_pursuit/path/waypoints_oncar.csv");
        this->declare_parameter("max_centerline_dist", 0.3);

        
        // ── NEW: Temporal tracking ──────────────────────────────────────────
        this->declare_parameter("track_assoc_dist", 0.50);   // max distance to associate detection to existing track (m)
        this->declare_parameter("track_min_hits", 2);        // frames a track must be seen before being published
        this->declare_parameter("track_max_misses", 1);      // frames a track can go unseen before being dropped
        this->declare_parameter("track_ema_alpha", 0.4);     // smoothing for track position

        std::string scan_topic = this->get_parameter("scan_topic").as_string();
        map_frame = this->get_parameter("map_frame").as_string();
        obs_radius = this->get_parameter("obstacle_radius").as_double();
        min_range = this->get_parameter("min_range").as_double();
        max_range = this->get_parameter("max_range").as_double();
        fov_fraction = this->get_parameter("fov_fraction").as_double();
        gap_threshold = this->get_parameter("gap_threshold").as_double();
        min_seg_pts = this->get_parameter("min_segment_points").as_int();
        max_seg_pts = this->get_parameter("max_segment_points").as_int();
        max_ext_seg_pts = this->get_parameter("max_extended_segment_points").as_int();
        max_distance = this->get_parameter("max_distance").as_double();
        std::string waypoints_csv = "/home/nvidia/ros2_ws/src/roboracer-final-race/pure_pursuit/path/waypoints_oncar.csv";
        max_centerline_dist = this->get_parameter("max_centerline_dist").as_double();

        std::string map_yaml = "/home/nvidia/f1tenth_ws/final_race.yaml";
        wall_inflation_radius = 1.0;

        track_assoc_dist = this->get_parameter("track_assoc_dist").as_double();
        track_min_hits = this->get_parameter("track_min_hits").as_int();
        track_max_misses = this->get_parameter("track_max_misses").as_int();
        track_ema_alpha = this->get_parameter("track_ema_alpha").as_double();

        // Load waypoints
        waypoints = load_waypoints(waypoints_csv);
        if (!waypoints.empty()) {
            RCLCPP_INFO(this->get_logger(), "Loaded %zu waypoints, centerline filter: ±%.2fm", waypoints.size(), max_centerline_dist);
        } else {
            RCLCPP_WARN(this->get_logger(), "No waypoints CSV — centerline filter disabled");
        }

        // Load occupancy map
        if (!map_yaml.empty()) {
            if (load_map(map_yaml, occ_map)) {
                RCLCPP_INFO(this->get_logger(),
                            "Loaded map %s (%dx%d, res=%.3f m/px), wall inflation: %.2fm",
                            map_yaml.c_str(), occ_map.width, occ_map.height,
                            occ_map.resolution, wall_inflation_radius);
            } else {
                RCLCPP_WARN(this->get_logger(), "Failed to load map %s — wall filter disabled", map_yaml.c_str());
            }
        } else {
            RCLCPP_WARN(this->get_logger(), "No map_yaml — wall filter disabled");
        }

        RCLCPP_INFO(this->get_logger(),
                    "Tracker: assoc_dist=%.2fm, min_hits=%d, max_misses=%d",
                    track_assoc_dist, track_min_hits, track_max_misses);

        // ── TF ───────────────────────────────────────────────────────────────
        tf_buffer = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener = std::make_unique<tf2_ros::TransformListener>(*tf_buffer);

        // ── Subscribers / Publishers ─────────────────────────────────────────
        scan_sub = this->create_subscription<sensor_msgs::msg::LaserScan>(
            scan_topic, 10, std::bind(&LidarObstacleNode::scan_callback, this, std::placeholders::_1));

        centroid_pub = this->create_publisher<geometry_msgs::msg::PoseArray>("/obstacles/centroids", 10);
        marker_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>("/obstacles/markers", 10);

        RCLCPP_INFO(this->get_logger(), "LidarObstacleNode ready — topic=%s, gap=%.2fm, seg_pts=[%d,%d], frame=%s",
                    scan_topic.c_str(), gap_threshold, min_seg_pts, max_seg_pts, map_frame.c_str());
    }

private:
    // ── Params ──────────────────────────────────────────────────────────────
    std::string map_frame;
    double obs_radius;
    double min_range;
    double max_range;
    double fov_fraction;
    double gap_threshold;
    int min_seg_pts;
    int max_seg_pts;
    int max_ext_seg_pts;
    double max_distance;
    double max_centerline_dist;

    double wall_inflation_radius;

    double track_assoc_dist;
    int track_min_hits;
    int track_max_misses;
    double track_ema_alpha;

    // ── State ───────────────────────────────────────────────────────────────
    std::unique_ptr<tf2_ros::Buffer> tf_buffer;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener;

    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr centroid_pub;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub;

    std::vector<std::pair<double, double>> waypoints;
    OccupancyMap occ_map;

    std::vector<ObstacleTrack> tracks_;
    int next_track_id_;

    // ── Waypoint loader (unchanged) ─────────────────────────────────────────
    std::vector<std::pair<double, double>> load_waypoints(const std::string& csv_path)
    {
        std::vector<std::pair<double, double>> pts;
        if (csv_path.empty()) return pts;

        std::ifstream file(csv_path);
        if (!file.is_open()) return pts;

        std::string line;
        bool is_header = true;

        while (std::getline(file, line)) {
            if (is_header) {
                is_header = false;
                continue;
            }
            std::stringstream ss(line);
            std::string v1_str, v2_str;
            if (std::getline(ss, v1_str, ',') && std::getline(ss, v2_str, ',')) {
                try {
                    double x = std::stod(v1_str);
                    double y = std::stod(v2_str);
                    pts.push_back({x, y});
                } catch (...) {
                }
            }
        }
        return pts;
    }

    // ── NEW: Map loader ─────────────────────────────────────────────────────
    // Parses a minimal ROS map_server-style YAML (looks for `image:`, `resolution:`,
    // `origin:`, `negate:`, `occupied_thresh:`, `free_thresh:`) and the referenced PGM.
    // No external YAML dep — just line-based parsing. Good enough for f1tenth-style maps.
    bool load_map(const std::string& yaml_path, OccupancyMap& out)
    {
        std::ifstream yf(yaml_path);
        if (!yf.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "Cannot open map yaml: %s", yaml_path.c_str());
            return false;
        }

        std::string image_rel;
        double resolution = 0.05;
        double origin_x = 0.0, origin_y = 0.0;
        int negate = 0;
        double occupied_thresh = 0.65;

        std::string line;
        while (std::getline(yf, line)) {
            // Strip comments
            auto hash = line.find('#');
            if (hash != std::string::npos) line = line.substr(0, hash);

            auto colon = line.find(':');
            if (colon == std::string::npos) continue;
            std::string key = line.substr(0, colon);
            std::string val = line.substr(colon + 1);

            // trim
            auto trim = [](std::string& s) {
                size_t a = s.find_first_not_of(" \t\r\n");
                size_t b = s.find_last_not_of(" \t\r\n");
                if (a == std::string::npos) { s.clear(); return; }
                s = s.substr(a, b - a + 1);
            };
            trim(key);
            trim(val);

            if (key == "image") {
                image_rel = val;
            } else if (key == "resolution") {
                resolution = std::stod(val);
            } else if (key == "origin") {
                // val looks like: [x, y, theta]
                auto lb = val.find('[');
                auto rb = val.find(']');
                if (lb != std::string::npos && rb != std::string::npos) {
                    std::string inner = val.substr(lb + 1, rb - lb - 1);
                    std::stringstream ss(inner);
                    std::string tok;
                    std::vector<double> nums;
                    while (std::getline(ss, tok, ',')) {
                        try { nums.push_back(std::stod(tok)); } catch (...) {}
                    }
                    if (nums.size() >= 2) {
                        origin_x = nums[0];
                        origin_y = nums[1];
                    }
                }
            } else if (key == "negate") {
                negate = std::stoi(val);
            } else if (key == "occupied_thresh") {
                occupied_thresh = std::stod(val);
            }
        }
        yf.close();

        if (image_rel.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Map yaml missing `image:` field");
            return false;
        }

        // Resolve image path relative to yaml directory
        std::string image_path = image_rel;
        if (image_rel.front() != '/') {
            auto slash = yaml_path.find_last_of('/');
            std::string dir = (slash == std::string::npos) ? "." : yaml_path.substr(0, slash);
            image_path = dir + "/" + image_rel;
        }

        // ── Parse PGM (P5 binary) ───────────────────────────────────────────
        std::ifstream pf(image_path, std::ios::binary);
        if (!pf.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "Cannot open map image: %s", image_path.c_str());
            return false;
        }

        std::string magic;
        pf >> magic;
        if (magic != "P5") {
            RCLCPP_ERROR(this->get_logger(), "Map image must be PGM P5 binary, got %s", magic.c_str());
            return false;
        }

        // Skip comments and read width/height/maxval
        auto skip_pgm_ws = [](std::ifstream& f) {
            int c;
            while (true) {
                c = f.peek();
                if (c == '#') {
                    std::string junk;
                    std::getline(f, junk);
                } else if (std::isspace(c)) {
                    f.get();
                } else {
                    break;
                }
            }
        };

        int W, H, maxval;
        skip_pgm_ws(pf); pf >> W;
        skip_pgm_ws(pf); pf >> H;
        skip_pgm_ws(pf); pf >> maxval;
        pf.get(); // single whitespace before pixel data

        if (W <= 0 || H <= 0 || maxval <= 0 || maxval > 255) {
            RCLCPP_ERROR(this->get_logger(), "Bad PGM header: %dx%d maxval=%d", W, H, maxval);
            return false;
        }

        std::vector<uint8_t> pgm(static_cast<size_t>(W) * H);
        pf.read(reinterpret_cast<char*>(pgm.data()), pgm.size());
        if (!pf) {
            RCLCPP_ERROR(this->get_logger(), "Failed to read PGM pixel data");
            return false;
        }
        pf.close();

        // Convert to occupancy. ROS convention: in the PGM, 0=black=occupied, 255=white=free.
        // We map to 0..100 where >occ_thresh means occupied. PGM is stored top-to-bottom,
        // ROS map (0,0) is bottom-left, so flip rows.
        out.width = W;
        out.height = H;
        out.resolution = resolution;
        out.origin_x = origin_x;
        out.origin_y = origin_y;
        out.occ_thresh = static_cast<int>(occupied_thresh * 100.0);
        out.data.assign(static_cast<size_t>(W) * H, 0);

        for (int row = 0; row < H; ++row) {
            for (int col = 0; col < W; ++col) {
                uint8_t px = pgm[row * W + col];
                double p;
                if (negate) {
                    p = static_cast<double>(px) / maxval;
                } else {
                    p = static_cast<double>(maxval - px) / maxval;
                }
                int occ = static_cast<int>(std::round(p * 100.0));
                int flipped_row = H - 1 - row;
                out.data[static_cast<size_t>(flipped_row) * W + col] = static_cast<uint8_t>(occ);
            }
        }

        out.loaded = true;

        // Pre-inflate occupied cells by wall_inflation_radius so runtime lookup is O(1).
        inflate_occupancy(out, wall_inflation_radius);

        return true;
    }

    // Dilate occupied cells outward by `radius_m`. After this, any cell whose
    // original-map distance to an occupied cell is <= radius_m is itself marked
    // occupied. We use a simple disc stamp — fine for one-time startup work.
    void inflate_occupancy(OccupancyMap& m, double radius_m)
    {
        if (radius_m <= 0.0) return;
        int radius_cells = static_cast<int>(std::ceil(radius_m / m.resolution));
        if (radius_cells <= 0) return;

        // Snapshot original occupied cells (so we don't keep growing as we stamp)
        std::vector<uint8_t> orig = m.data;

        // Precompute disc offset list
        std::vector<std::pair<int,int>> disc;
        double r2 = radius_m * radius_m;
        for (int dj = -radius_cells; dj <= radius_cells; ++dj) {
            for (int di = -radius_cells; di <= radius_cells; ++di) {
                double dx = di * m.resolution;
                double dy = dj * m.resolution;
                if (dx * dx + dy * dy <= r2) {
                    disc.push_back({di, dj});
                }
            }
        }

        size_t inflated_count = 0;
        for (int j = 0; j < m.height; ++j) {
            for (int i = 0; i < m.width; ++i) {
                if (orig[static_cast<size_t>(j) * m.width + i] > m.occ_thresh) {
                    for (const auto& off : disc) {
                        int ni = i + off.first;
                        int nj = j + off.second;
                        if (ni < 0 || nj < 0 || ni >= m.width || nj >= m.height) continue;
                        size_t idx = static_cast<size_t>(nj) * m.width + ni;
                        if (m.data[idx] <= m.occ_thresh) {
                            m.data[idx] = 100;
                            ++inflated_count;
                        }
                    }
                }
            }
        }
        RCLCPP_INFO(this->get_logger(),
                    "Inflated map by %.2fm (%d cells) — added %zu occupied cells",
                    radius_m, radius_cells, inflated_count);
    }

    // Single-cell lookup against the pre-inflated map.
    bool is_near_wall(double mx, double my) const
    {
        if (!occ_map.loaded) return false;

        int i = static_cast<int>(std::floor((mx - occ_map.origin_x) / occ_map.resolution));
        int j = static_cast<int>(std::floor((my - occ_map.origin_y) / occ_map.resolution));

        // Out-of-bounds points are treated as non-wall (they can't be inside the map's walls).
        if (i < 0 || j < 0 || i >= occ_map.width || j >= occ_map.height) return false;

        return occ_map.data[static_cast<size_t>(j) * occ_map.width + i] > occ_map.occ_thresh;
    }

    // ── NEW: Tracker update ─────────────────────────────────────────────────
    // Greedy nearest-neighbor association. Detections that don't match an existing
    // track create a new tentative track. Tracks aged out after `track_max_misses`.
    // Only confirmed tracks (>= track_min_hits) are returned for publication.
    std::vector<std::pair<double, double>> update_tracks(
        const std::vector<std::pair<double, double>>& detections)
    {
        std::vector<bool> det_used(detections.size(), false);
        std::vector<bool> track_updated(tracks_.size(), false);

        // For each existing track, find closest unused detection within gate
        for (size_t t = 0; t < tracks_.size(); ++t) {
            double best_d2 = track_assoc_dist * track_assoc_dist;
            int best_d = -1;
            for (size_t d = 0; d < detections.size(); ++d) {
                if (det_used[d]) continue;
                double dx = detections[d].first - tracks_[t].x;
                double dy = detections[d].second - tracks_[t].y;
                double d2 = dx * dx + dy * dy;
                if (d2 <= best_d2) {
                    best_d2 = d2;
                    best_d = static_cast<int>(d);
                }
            }
            if (best_d >= 0) {
                // EMA update
                double a = track_ema_alpha;
                tracks_[t].x = (1.0 - a) * tracks_[t].x + a * detections[best_d].first;
                tracks_[t].y = (1.0 - a) * tracks_[t].y + a * detections[best_d].second;
                tracks_[t].hits += 1;
                tracks_[t].misses = 0;
                if (tracks_[t].hits >= track_min_hits) {
                    tracks_[t].confirmed = true;
                }
                det_used[best_d] = true;
                track_updated[t] = true;
            }
        }

        // Unmatched tracks → increment misses
        for (size_t t = 0; t < tracks_.size(); ++t) {
            if (!track_updated[t]) {
                tracks_[t].misses += 1;
            }
        }

        // Drop expired tracks
        tracks_.erase(
            std::remove_if(tracks_.begin(), tracks_.end(),
                [this](const ObstacleTrack& tr) { return tr.misses > track_max_misses; }),
            tracks_.end()
        );

        // Spawn new tentative tracks from unmatched detections
        for (size_t d = 0; d < detections.size(); ++d) {
            if (det_used[d]) continue;
            ObstacleTrack tr;
            tr.id = next_track_id_++;
            tr.x = detections[d].first;
            tr.y = detections[d].second;
            tr.hits = 1;
            tr.misses = 0;
            tr.confirmed = (track_min_hits <= 1);
            tracks_.push_back(tr);
        }

        // Return only confirmed tracks
        std::vector<std::pair<double, double>> out;
        for (const auto& tr : tracks_) {
            if (tr.confirmed) {
                out.push_back({tr.x, tr.y});
            }
        }
        return out;
    }

    void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
    {
        size_t num = msg->ranges.size();
        double half_arc = (msg->angle_max - msg->angle_min) * fov_fraction;

        // ── Stage 1: range/FOV gate ─────────────────────────────────────────
        std::vector<float> valid_ranges;
        std::vector<float> valid_angles;
        valid_ranges.reserve(num);
        valid_angles.reserve(num);

        for (size_t i = 0; i < num; ++i) {
            float angle = msg->angle_min + i * msg->angle_increment;
            float r = msg->ranges[i];
            if (std::isfinite(r) &&
                (angle >= -half_arc) && (angle <= half_arc) &&
                (r >= std::max(msg->range_min, static_cast<float>(min_range))) &&
                (r <= std::min(msg->range_max, static_cast<float>(max_range))))
            {
                valid_ranges.push_back(r);
                valid_angles.push_back(angle);
            }
        }

        // Helper: age tracks even when we have no detections this frame
        auto publish_with_aging = [&](const std::vector<std::pair<double, double>>& dets) {
            auto confirmed = update_tracks(dets);
            std_msgs::msg::Header header;
            header.stamp = msg->header.stamp;
            header.frame_id = map_frame;
            publish_centroids(header, confirmed);
            publish_markers(header, confirmed);
        };

        if (valid_ranges.size() < static_cast<size_t>(min_seg_pts)) {
            publish_with_aging({});
            return;
        }

        // ── Stage 2: TF lookup (laser → map) ────────────────────────────────
        geometry_msgs::msg::TransformStamped tf;
        try {
            tf = tf_buffer->lookupTransform(
                map_frame,
                msg->header.frame_id,
                tf2::TimePointZero,
                tf2::durationFromSec(0.05)
            );
        } catch (const tf2::TransformException& ex) {
            RCLCPP_WARN(this->get_logger(), "TF lookup failed: %s", ex.what());
            return;
        }

        // ── Stage 3: transform every point to map frame and drop wall hits ──
        // Points are kept in scan order so gap segmentation stays meaningful.
        // We keep both laser-frame coords (for distance-based segment threshold)
        // and map-frame coords (for centroid output and segmentation gaps).
        size_t valid_num = valid_ranges.size();
        std::vector<double> x_laser, y_laser, x_map, y_map;
        x_laser.reserve(valid_num);
        y_laser.reserve(valid_num);
        x_map.reserve(valid_num);
        y_map.reserve(valid_num);

        size_t wall_dropped = 0;
        for (size_t i = 0; i < valid_num; ++i) {
            double xl = valid_ranges[i] * std::cos(valid_angles[i]);
            double yl = valid_ranges[i] * std::sin(valid_angles[i]);
            double xm, ym;
            transform_point_2d(xl, yl, tf, xm, ym);

            if (is_near_wall(xm, ym)) {
                ++wall_dropped;
                continue;
            }
            x_laser.push_back(xl);
            y_laser.push_back(yl);
            x_map.push_back(xm);
            y_map.push_back(ym);
        }

        size_t kept_num = x_map.size();
        if (kept_num < static_cast<size_t>(min_seg_pts)) {
            publish_with_aging({});
            return;
        }

        // ── Stage 4: gap segmentation on the kept points ────────────────────
        // Note: we segment in map frame here. Adjacent kept points that used
        // to be neighbors in the scan may no longer be spatially adjacent if
        // wall points were removed between them — that's correct, the gap
        // threshold will naturally split those into separate segments.
        std::vector<size_t> split_indices;
        for (size_t i = 0; i + 1 < kept_num; ++i) {
            double dx = x_map[i+1] - x_map[i];
            double dy = y_map[i+1] - y_map[i];
            if (std::hypot(dx, dy) > gap_threshold) {
                split_indices.push_back(i + 1);
            }
        }

        std::vector<size_t> starts;
        std::vector<size_t> ends;
        starts.push_back(0);
        for (size_t idx : split_indices) starts.push_back(idx);
        for (size_t idx : split_indices) ends.push_back(idx);
        ends.push_back(kept_num);

        // ── Stage 5: per-segment centroid + remaining filters ───────────────
        std::vector<std::pair<double, double>> detections;
        for (size_t j = 0; j < starts.size(); ++j) {
            size_t s = starts[j];
            size_t e = ends[j];
            size_t seg_len = e - s;

            if (seg_len < static_cast<size_t>(min_seg_pts)) continue;

            double sum_xl = 0.0, sum_yl = 0.0, sum_xm = 0.0, sum_ym = 0.0;
            for (size_t k = s; k < e; ++k) {
                sum_xl += x_laser[k];
                sum_yl += y_laser[k];
                sum_xm += x_map[k];
                sum_ym += y_map[k];
            }
            double cx_laser = sum_xl / seg_len;
            double cy_laser = sum_yl / seg_len;
            double cx_map = sum_xm / seg_len;
            double cy_map = sum_ym / seg_len;

            // Distance-scaled max-segment-points (uses laser-frame distance from car)
            double distance = std::hypot(cx_laser, cy_laser);
            double clamped_distance = std::max(0.0, std::min(distance, max_distance));
            double distance_ratio = 1.0 - (clamped_distance / max_distance);
            double dynamic_max_seg_pts = max_seg_pts + (max_ext_seg_pts - max_seg_pts) * distance_ratio;
            if (seg_len > static_cast<size_t>(dynamic_max_seg_pts)) continue;

            // Centerline filter
            // if (!waypoints.empty()) {
            //     double min_d = 1e9;
            //     for (const auto& wp : waypoints) {
            //         double dx = wp.first - cx_map;
            //         double dy = wp.second - cy_map;
            //         double d = std::hypot(dx, dy);
            //         if (d < min_d) min_d = d;
            //     }
            //     if (min_d > max_centerline_dist) continue;
            // }

            detections.push_back({cx_map, cy_map});
        }

        // ── Stage 6: temporal tracking ──────────────────────────────────────
        publish_with_aging(detections);

        RCLCPP_DEBUG(this->get_logger(),
                     "valid=%zu, wall_dropped=%zu, kept=%zu, raw_dets=%zu, tracks=%zu",
                     valid_num, wall_dropped, kept_num, detections.size(), tracks_.size());
    }

    void transform_point_2d(double x, double y, const geometry_msgs::msg::TransformStamped& tf, double& mx, double& my)
    {
        auto t = tf.transform.translation;
        auto q = tf.transform.rotation;

        double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
        double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
        double yaw = std::atan2(siny_cosp, cosy_cosp);

        double cos_y = std::cos(yaw);
        double sin_y = std::sin(yaw);

        mx = cos_y * x - sin_y * y + t.x;
        my = sin_y * x + cos_y * y + t.y;
    }

    void publish_centroids(const std_msgs::msg::Header& header, const std::vector<std::pair<double, double>>& centroids)
    {
        geometry_msgs::msg::PoseArray msg;
        msg.header = header;
        for (const auto& c : centroids) {
            geometry_msgs::msg::Pose p;
            p.position.x = c.first;
            p.position.y = c.second;
            p.position.z = 0.0;
            p.orientation.w = 1.0;
            msg.poses.push_back(p);
        }
        centroid_pub->publish(msg);
    }

    void publish_markers(const std_msgs::msg::Header& header, const std::vector<std::pair<double, double>>& centroids)
    {
        visualization_msgs::msg::MarkerArray ma;

        visualization_msgs::msg::Marker delete_all;
        delete_all.header = header;
        delete_all.ns = "obstacles";
        delete_all.action = visualization_msgs::msg::Marker::DELETEALL;
        ma.markers.push_back(delete_all);

        builtin_interfaces::msg::Duration lifetime;
        lifetime.sec = 0;
        lifetime.nanosec = 133333333;

        for (size_t i = 0; i < centroids.size(); ++i) {
            visualization_msgs::msg::Marker dot;
            dot.header = header;
            dot.ns = "obstacle_centroid";
            dot.id = static_cast<int>(i);
            dot.type = visualization_msgs::msg::Marker::SPHERE;
            dot.action = visualization_msgs::msg::Marker::ADD;
            dot.pose.position.x = centroids[i].first;
            dot.pose.position.y = centroids[i].second;
            dot.pose.position.z = 0.0;
            dot.pose.orientation.w = 1.0;
            dot.scale.x = 0.15;
            dot.scale.y = 0.15;
            dot.scale.z = 0.15;
            dot.color.r = 1.0;
            dot.color.g = 0.2;
            dot.color.b = 0.2;
            dot.color.a = 1.0;
            dot.lifetime = lifetime;
            ma.markers.push_back(dot);

            visualization_msgs::msg::Marker rs;
            rs.header = header;
            rs.ns = "obstacle_radius";
            rs.id = static_cast<int>(i);
            rs.type = visualization_msgs::msg::Marker::SPHERE;
            rs.action = visualization_msgs::msg::Marker::ADD;
            rs.pose.position.x = centroids[i].first;
            rs.pose.position.y = centroids[i].second;
            rs.pose.position.z = 0.0;
            rs.pose.orientation.w = 1.0;
            double d = obs_radius * 2.0;
            rs.scale.x = d;
            rs.scale.y = d;
            rs.scale.z = d;
            rs.color.r = 1.0;
            rs.color.g = 0.4;
            rs.color.b = 0.0;
            rs.color.a = 0.2;
            rs.lifetime = lifetime;
            ma.markers.push_back(rs);
        }

        marker_pub->publish(ma);
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<LidarObstacleNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
