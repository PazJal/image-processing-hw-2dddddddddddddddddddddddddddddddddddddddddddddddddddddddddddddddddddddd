import cv2
import matplotlib.pyplot as plt
import numpy as np
# size of the image
m,n = 921, 750

# frame points of the blank wormhole image
src_points = np.float32([[0, 0],
                            [int(n / 3), 0],
                            [int(2 * n /3), 0],
                            [n, 0],
                            [n, m],
                            [int(2 * n / 3), m],
                            [int(n / 3), m],
                            [0, m]])

# blank wormhole frame points
dst_points = np.float32([[96, 282],
                       [220, 276],
                       [344, 276],
                       [468, 282],
                       [474, 710],
                       [350, 744],
                       [227, 742],
                       [103, 714]]
                      )

#Test Matrix
test = np.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]
)

def find_transform(pointset1, pointset2):

    # create the pseudo inverse from the pointsets
    pinv = get_psuedo_invert(pointset1,pointset2)

    # create a column vector from the target pointset. 
    vec = pointset1.reshape((16,1))

    # caculate the coefficient matrix. 
    coefficentMatrix = np.matmul(pinv, vec)

    #manually create the transformation matrix:
    transformation_matrix = np.zeros((3,3))

    transformation_matrix[0][0] = coefficentMatrix[0]
    transformation_matrix[0][1] = coefficentMatrix[1]
    transformation_matrix[0][2] = coefficentMatrix[4]
    transformation_matrix[1][0] = coefficentMatrix[2]
    transformation_matrix[1][1] = coefficentMatrix[3]
    transformation_matrix[1][2] = coefficentMatrix[5]
    transformation_matrix[2][0] = coefficentMatrix[6]
    transformation_matrix[2][1] = coefficentMatrix[7]
    transformation_matrix[2][2] = 1

    # setting the laset entry to 1. 
    transformation_matrix[2][2] = 1

    return transformation_matrix


def transform_image(image, T):
    #Get width and height.
    (image_height, image_width) = image.shape

    #Initialize new image
    new_image = np.zeros(image.shape)

    for w in range(0,image_width):
        for h in range(0, image_height):
            # create a vector for the coordinates
            sourceVector = np.matmul(T,np.array([w, h, 1 ]).transpose())
            
            # normalize the vector
            normSourceVector = (sourceVector / sourceVector[2])
            
            # destructre the coordinates
            x = normSourceVector.item(0)
            y = normSourceVector.item(1)
            xSrc = int(round(x))
            ySrc = int(round(y))
            
            # handle source coordinates that are not in bound
            if ySrc >= image_height or ySrc < 0:
                new_image[h][w] = image[h][w]
            elif xSrc >= image_width or xSrc < 0:
                new_image[h][w] = image[h][w]
            else:
                new_image[h][w] = image[ySrc][xSrc]

    return new_image


def create_wormhole(im, T, iter=5):
    # Transform the image
    new_image = transform_image(im, T)

    # recurse if needed
    if iter == 1 :
        return new_image
    else: 
        return create_wormhole(new_image, T / 2, iter - 1)

def get_psuedo_invert(target_points, source_points):
    
    # initialize transformation matrix. 
    t = np.zeros((target_points.size, 8))

    # initialize row counter
    r = 0

    # build two rows per point. 
    for sPoint,tPoint in zip(source_points,target_points):

        # destructre the coordinates
        x = sPoint[0]
        y = sPoint[1]
        xTag = tPoint[0]
        yTag = tPoint[1]

        # build first row
        t[2 * r] = np.array([x, y, 0, 0, 1, 0, -x * xTag, -y * xTag])

        # build second row
        t[2 * r + 1] = np.array([0, 0, x, y, 0, 1, -x *yTag, -y * yTag])
        
        r += 1
    
    # destructre the matrix computation
    mul = np.matmul(t.transpose(),  t)
    invers = np.linalg.inv(mul)
    pinv = np.matmul( invers , t.transpose())

    return pinv