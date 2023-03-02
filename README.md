# geometry #

Tools for handling 3D geometry, mostly just adds a nice interface to various geometric enterties. Each geometric entity can also be a vector of geometric entities. Each entity wraps a numpy array with the relevant number of columns labelled according to the cols class property and rows equal to the number of elements in the vector. Attribute access to each column is available and returns a numpy array. 

Where operations are supported between geometric types the size of the output is inferred based on the length of the inputs. Where the two vectors of entities are of the same length, elementwise operations are performed. Where one vector is length one and the other is greater than one then the operation will be performed on every element of the longer vector. 

Magic methods are used extensively and the function of operators are logical for each type. If unsure what the logical option is then check the code where it should be pretty clear. 

Many convenience methods and constructors are available. Documentation is limited but if you need something it has probably already been written so check the code first. 


now available on pypi:

pip install pfc-geometry