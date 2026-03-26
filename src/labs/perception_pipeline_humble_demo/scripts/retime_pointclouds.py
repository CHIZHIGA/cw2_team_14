#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2


class PointCloudRetimer(Node):
    def __init__(self) -> None:
        super().__init__("pointcloud_retimer")
        self.pub1 = self.create_publisher(PointCloud2, "/camera_1/points_retimed", 10)
        self.pub2 = self.create_publisher(PointCloud2, "/camera_2/points_retimed", 10)
        self.sub1 = self.create_subscription(
            PointCloud2, "/camera_1/points", self.cb1, 10
        )
        self.sub2 = self.create_subscription(
            PointCloud2, "/camera_2/points", self.cb2, 10
        )

    def _retime_and_publish(self, msg: PointCloud2, pub) -> None:
        msg.header.stamp = self.get_clock().now().to_msg()
        pub.publish(msg)

    def cb1(self, msg: PointCloud2) -> None:
        self._retime_and_publish(msg, self.pub1)

    def cb2(self, msg: PointCloud2) -> None:
        self._retime_and_publish(msg, self.pub2)


def main() -> None:
    rclpy.init()
    node = PointCloudRetimer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
