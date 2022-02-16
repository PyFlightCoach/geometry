from geometry import Point, Quaternion, Transformation



class Circle():
    def __init__(self, radius: float, transform: Transformation):
        """The circle is about the Z axis, centre on the origin. The transformation 
        defines the location of the axis.

        Args:
            radius (float): The radius of the circle
            transform (Transformation): The transformation to the circles axis (0,0,1)
        """
        self.radius = radius
        self.transform = transform
    
#change