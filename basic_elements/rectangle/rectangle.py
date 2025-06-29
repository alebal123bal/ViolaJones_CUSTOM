"""
Simple class representing a rectangle.
"""


class Rectangle:
    """
    Simple class representing a rectangle.
    """

    def __init__(self, x, y, width, height, sign):
        """
        Initialize a Rectangle.

        Args:
            x (int): Top-left x coordinate
            y (int): Top-left y coordinate
            width (int): Width of the rectangle
            height (int): Height of the rectangle
            sign (int): Sign value, should be +1 or -1
        """

        self.x = x
        self.y = y
        self.width = width
        self.height = height

        if sign not in [-1, 1]:
            raise ValueError("Sign must be either +1 or -1")
        self.sign = sign

    @property
    def x2(self):
        """Right edge x coordinate"""
        return self.x + self.width

    @property
    def y2(self):
        """Bottom edge y coordinate"""
        return self.y + self.height

    def area(self):
        """Calculate the area of the rectangle"""
        return self.width * self.height

    def scale(self, scale):
        """
        Scale the rectangle by a factor.

        Args:
            scale (float): Scale factor to resize the rectangle
        """

        if scale <= 0:
            raise ValueError("Scale factor must be positive")

        self.x = int(self.x * scale)
        self.y = int(self.y * scale)
        self.width = int(self.width * scale)
        self.height = int(self.height * scale)

    def __repr__(self):
        return f"""x={self.x}, y={self.y}, w={self.width}, h={self.height}, sign={self.sign:+d}"""

    def __eq__(self, other):
        if not isinstance(other, Rectangle):
            return False
        return (
            self.x == other.x
            and self.y == other.y
            and self.width == other.width
            and self.height == other.height
            and self.sign == other.sign
        )
