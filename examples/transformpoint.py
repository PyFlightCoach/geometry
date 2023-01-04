from geometry import Point, Transformation, Coord, P0, PX, PY, PZ

#note all the geometry objects hold vectors of whatever they represent. 
# if the lengths of the vectors are the same it does elementwise operations
#if one is length one and the other >1 then it'll do the same operation to all items in the longer one
# if both lengths are >1 but not the same then it'll fall over 


#create two coordinate frames
c1 = Coord.from_xy(P0(), PY(), PZ()) #P0, PX, PY, PZ are helper functions to create point objects
c2 = Coord.from_nothing()

#create the transformation
transform = Transformation.from_coords(c1, c2)

#point to be transformed
pin = Point(10, 20, 30)

#transfrom the point
po = transform.apply(pin)

print(po)