import rclpy
from rclpy.node import Node


class HelloNode(Node):

    def __init__(self):
        super().__init__("hello_node")

        self.timer = self.create_timer(
            1.0,
            self.say_hello
        )

    def say_hello(self):
        self.get_logger().info("Hello from ROS 2!")


def main(args=None):
    rclpy.init(args=args)

    node = HelloNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()