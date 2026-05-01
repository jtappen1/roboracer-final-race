#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>

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

class LidarObstacleNode : public rclcpp::Node
{
public:
    LidarObstacleNode() : Node("lidar_obstacle_node")
    {
        // ── Parameters ──────────────────────────────────────────────────────
        this->declare_parameter("scan_topic", "/scan");
        this->declare_parameter("map_frame", "map");
        this->declare_parameter("obstacle_radius", 0.5); // sphere radius for visualisation (m)

        // Scan filtering
        this->declare_parameter("min_range", 0.1); // ignore returns closer than this (m)
        this->declare_parameter("max_range", 5.0); // ignore returns beyond this (m)
        this->declare_parameter("fov_fraction", 0.1); // fraction of scan arc each side of fwd

        // Gap segmentation
        this->declare_parameter("gap_threshold", 0.10); // >10cm between adjacent pts = new segment
        this->declare_parameter("min_segment_points", 5); // ignore tiny noise segments
        this->declare_parameter("max_segment_points", 50); // ignore large wall segments

        // Distance-based threshold parameters
        this->declare_parameter("max_extended_segment_points", 150); // max points allowed at close range
        this->declare_parameter("max_distance", 5.0); // max distance for scaling

        // Centerline filter
        this->declare_parameter("waypoints_csv", "/home/jtappen/roboracer_ws/src/final-race/pure_pursuit/path/levine_2floor_points.csv");
        this->declare_parameter("max_centerline_dist", 0.5); // reject centroids further than this (m)

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
        std::string waypoints_csv = this->get_parameter("waypoints_csv").as_string();
        max_centerline_dist = this->get_parameter("max_centerline_dist").as_double();

        // Load waypoints
        waypoints = load_waypoints(waypoints_csv);
        if (!waypoints.empty()) {
            RCLCPP_INFO(this->get_logger(), "Loaded %zu waypoints, centerline filter: ±%.2fm", waypoints.size(), max_centerline_dist);
        } else {
            RCLCPP_WARN(this->get_logger(), "No waypoints CSV — centerline filter disabled");
        }

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

    std::unique_ptr<tf2_ros::Buffer> tf_buffer;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener;

    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr centroid_pub;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub;

    std::vector<std::pair<double, double>> waypoints;

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
                    // Skip invalid lines
                }
            }
        }
        return pts;
    }

    void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
    {
        size_t num = msg->ranges.size();
        std::vector<float> ranges(num);
        std::vector<float> angles(num);

        for (size_t i = 0; i < num; ++i) {
            angles[i] = msg->angle_min + i * msg->angle_increment;
            ranges[i] = msg->ranges[i];
        }

        double half_arc = (msg->angle_max - msg->angle_min) * fov_fraction;

        // Forward FOV gate
        std::vector<float> valid_ranges;
        std::vector<float> valid_angles;
        for (size_t i = 0; i < num; ++i) {
            if (std::isfinite(ranges[i]) &&
                (angles[i] >= -half_arc) && (angles[i] <= half_arc) &&
                (ranges[i] >= std::max(msg->range_min, static_cast<float>(min_range))) &&
                (ranges[i] <= std::min(msg->range_max, static_cast<float>(max_range))))
            {
                valid_ranges.push_back(ranges[i]);
                valid_angles.push_back(angles[i]);
            }
        }

        if (valid_ranges.size() < static_cast<size_t>(min_seg_pts)) {
            publish_empty(msg->header.stamp);
            return;
        }

        // TF lookup — laser → map
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

        size_t valid_num = valid_ranges.size();
        std::vector<double> x_laser(valid_num);
        std::vector<double> y_laser(valid_num);

        for (size_t i = 0; i < valid_num; ++i) {
            x_laser[i] = valid_ranges[i] * std::cos(valid_angles[i]);
            y_laser[i] = valid_ranges[i] * std::sin(valid_angles[i]);
        }

        // Gap segmentation in laser frame
        std::vector<size_t> split_indices;
        for (size_t i = 0; i < valid_num - 1; ++i) {
            double dx = x_laser[i+1] - x_laser[i];
            double dy = y_laser[i+1] - y_laser[i];
            double dist = std::hypot(dx, dy);
            if (dist > gap_threshold) {
                split_indices.push_back(i + 1);
            }
        }

        std::vector<size_t> starts;
        std::vector<size_t> ends;

        starts.push_back(0);
        for (size_t idx : split_indices) {
            starts.push_back(idx);
        }

        for (size_t idx : split_indices) {
            ends.push_back(idx);
        }
        ends.push_back(valid_num);

        std::vector<std::pair<double, double>> centroids;
        for (size_t j = 0; j < starts.size(); ++j) {
            size_t s = starts[j];
            size_t e = ends[j];
            size_t seg_len = e - s;

            if (seg_len < static_cast<size_t>(min_seg_pts)) {
                continue;
            }

            double sum_x = 0.0, sum_y = 0.0;
            for (size_t k = s; k < e; ++k) {
                sum_x += x_laser[k];
                sum_y += y_laser[k];
            }
            double cx_laser = sum_x / seg_len;
            double cy_laser = sum_y / seg_len;

            // Dynamic calculation of max_segment_points based on distance to segment
            double distance = std::hypot(cx_laser, cy_laser);
            double clamped_distance = std::max(0.0, std::min(distance, max_distance));
            double distance_ratio = 1.0 - (clamped_distance / max_distance);
            double dynamic_max_seg_pts = max_seg_pts + (max_ext_seg_pts - max_seg_pts) * distance_ratio;

            if (seg_len > static_cast<size_t>(dynamic_max_seg_pts)) {
                continue;
            }

            double cx_map, cy_map;
            transform_point_2d(cx_laser, cy_laser, tf, cx_map, cy_map);

            // Centerline filter
            if (!waypoints.empty()) {
                double min_d = 1e9;
                for (const auto& wp : waypoints) {
                    double dx = wp.first - cx_map;
                    double dy = wp.second - cy_map;
                    double d = std::hypot(dx, dy);
                    if (d < min_d) {
                        min_d = d;
                    }
                }
                if (min_d > max_centerline_dist) {
                    continue;
                }
            }

            centroids.push_back({cx_map, cy_map});
        }

        std_msgs::msg::Header header;
        header.stamp = msg->header.stamp;
        header.frame_id = map_frame;

        publish_centroids(header, centroids);
        publish_markers(header, centroids);

        RCLCPP_DEBUG(this->get_logger(), "Found %zu obstacle(s)", centroids.size());
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

        // Clear previous frame
        visualization_msgs::msg::Marker delete_all;
        delete_all.header = header;
        delete_all.ns = "obstacles";
        delete_all.action = visualization_msgs::msg::Marker::DELETEALL;
        ma.markers.push_back(delete_all);

        builtin_interfaces::msg::Duration lifetime;
        lifetime.sec = 0;
        lifetime.nanosec = 133333333; // 2 lidar frames at 15Hz

        for (size_t i = 0; i < centroids.size(); ++i) {
            // Solid centroid dot
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

            // Transparent radius sphere
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

    void publish_empty(const builtin_interfaces::msg::Time& stamp)
    {
        std_msgs::msg::Header header;
        header.stamp = stamp;
        header.frame_id = map_frame;
        publish_centroids(header, {});
        publish_markers(header, {});
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