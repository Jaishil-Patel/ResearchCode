import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class GreetingPublisher(Node):

    def __init__(self):
        super().__init__("greeting_publisher")

        self.publisher = self.create_publisher(
            String,
            "greetings",
            10
        )

        self.counter = 0

        self.timer = self.create_timer(
            1.0,
            self.publish_message
        )

    def publish_message(self):
        msg = String()

        self.counter += 1

        msg.data = f"Hello {self.counter}"

        self.publisher.publish(msg)

        self.get_logger().info(f"Published: {msg.data}")


def main(args=None):
    rclpy.init(args=args)

    node = GreetingPublisher()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()