#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include <chrono>
#include <vector>
#include <numeric>
#include <std_msgs/msg/bool.hpp>
#include <algorithm>
#include <limits>
#include <cmath>

class Safety : public rclcpp::Node {

public:
    Safety() : Node("safety_node")
    {
        /*
        You should also subscribe to the /scan topic to get the
        sensor_msgs/LaserScan messages and the /ego_racecar/odom topic to get
        the nav_msgs/Odometry messages

        The subscribers should use the provided odom_callback and 
        scan_callback as callback methods

        NOTE that the x component of the linear velocity in odom is the speed
        */

        /// TODO: create ROS subscribers and publishers
        // parameter
        this->declare_parameter<double>("ttc_threshold", 0.5);
        this->get_parameter("ttc_threshold", ttc_threshold);

        // publisher: /drive
        drive_pub = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
            "/drive", 10);
        
        // subscriber: /ego_racecar/odom
        odom_sub = this->create_subscription<nav_msgs::msg::Odometry>(
            "/pf/pose/odom", 10,
            std::bind(&Safety::drive_callback, this, std::placeholders::_1));

        brake_pub = this->create_publisher<std_msgs::msg::Bool>("/aeb_brake", 10);

        // subscriber: /scan (sensor QoS)
        scan_sub = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", rclcpp::SensorDataQoS(),
            std::bind(&Safety::scan_callback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "C++ safety_node started");
        
    }

private:
    double speed = 0.0;
    double ttc_threshold = 1.0;
    /// TODO: create ROS subscribers and publishers
    std::vector<double> cb_times_ms_;
    size_t max_samples_ = 100;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr brake_pub;

    bool brake_active = false;
    int clear_count = 0;

    double min_stop_time = 1.0;      // at least stop for 1 s
    int clear_confirm_count = 10;    // 10 scans to clear the obstacle
    rclcpp::Time brake_start_time;
    void drive_callback(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
    {
        /// TODO: update current speed
        speed = msg->twist.twist.linear.x;
    }

    void scan_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg) 
    {
        /// TODO: calculate TTC

        /// TODO: publish drive/brake message
                // keep parameter up to date (optional, but handy when you set it at runtime)
        auto t0 = std::chrono::steady_clock::now(); // start time
        this->get_parameter("ttc_threshold", ttc_threshold);
        // 1. Extract ranges and angles from the scan_msg
        const auto & ranges = scan_msg->ranges;
        const size_t n = ranges.size();
        if (n == 0) return;

        const double inf = std::numeric_limits<double>::infinity();

        double min_ttc = inf;

        // iterate over beams
        for (size_t i = 0; i < n; ++i) {
            const float r = ranges[i];

            // valid range check
            if (!std::isfinite(r) || r <= 0.0f) continue;

            const double angle = scan_msg->angle_min + static_cast<double>(i) * scan_msg->angle_increment;

            // 2. Calculate r_dot
            const double r_dot = -speed * std::cos(angle);
            
            // 3. Calculate TTC
            // if r_dot < 0: iTTC = r / (-r_dot) else +inf
            double ttc = inf;
            if (r_dot < 0.0) {
                ttc = static_cast<double>(r) / (-r_dot);
            }

            if (ttc < min_ttc) min_ttc = ttc;
        }

        // 4. Threshold TTC and decide whether to brake
        bool danger = (speed > 0.1 && min_ttc < ttc_threshold);

        if (danger) {
            if (!brake_active) {
                brake_start_time = this->now();
                RCLCPP_WARN(this->get_logger(),
                    "[AEB START] TTC=%.2f < %.2f, braking!",
                    min_ttc, ttc_threshold);
            }

            brake_active = true;
            clear_count = 0;
        } else {
            if (brake_active) {
                double stopped_time = (this->now() - brake_start_time).seconds();

                if (stopped_time >= min_stop_time) {
                    clear_count++;

                    if (clear_count >= clear_confirm_count) {
                        brake_active = false;
                        clear_count = 0;

                        RCLCPP_INFO(this->get_logger(),
                            "[AEB CLEAR] obstacle clear, releasing brake");
                    }
                }
            }
        }

        // publish brake state every scan
        std_msgs::msg::Bool brake_msg;
        brake_msg.data = brake_active;
        brake_pub->publish(brake_msg);

        // optional: also publish direct stop command when braking
        if (brake_active) {
            ackermann_msgs::msg::AckermannDriveStamped drive_msg;
            drive_msg.header.stamp = this->now();
            drive_msg.drive.speed = 0.0;
            drive_msg.drive.steering_angle = 0.0;
            drive_pub->publish(drive_msg);

            RCLCPP_WARN(this->get_logger(),
                "[AEB STOP] TTC=%.2f | stopped=%s | clear_count=%d",
                min_ttc,
                brake_active ? "true" : "false",
                clear_count);
        }

        // end execution time measurement
        auto t1 = std::chrono::steady_clock::now();
        double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        cb_times_ms_.push_back(dt_ms);
        if (cb_times_ms_.size() >= max_samples_) {
            double avg = std::accumulate(cb_times_ms_.begin(), cb_times_ms_.end(), 0.0) / cb_times_ms_.size();
            double worst = *std::max_element(cb_times_ms_.begin(), cb_times_ms_.end());
            RCLCPP_INFO(this->get_logger(), "[scan_callback] avg=%.3f ms, worst=%.3f ms (N=%zu)",
                        avg, worst, cb_times_ms_.size());
            cb_times_ms_.clear();
        }
    }



};
int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Safety>());
    rclcpp::shutdown();
    return 0;
}
