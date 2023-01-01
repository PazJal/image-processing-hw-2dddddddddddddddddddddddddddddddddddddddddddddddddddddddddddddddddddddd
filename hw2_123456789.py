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
    pinv = get_psuedo_invert(pointset2,pointset1)

    # create a column vector from the target pointset. 
    vec = pointset2.reshape((16,1))

    # caculate the coefficient matrix. 
    coefficentMatrix = np.matmul(pinv, vec)

    # rearrage the coefficients into a matrix. 
    transformation_matrix = np.resize(coefficentMatrix,(3,3))

    # setting the laset entry to 1. 
    transformation_matrix[2][2] = 1


    # np.array(coefficentMatrix)
    #result = np.ones((3,3))
    # c = 0 
    # r = 0
    # for r in range (0,3):
    #     for c in range(0,3):
    #         if (r * 3 + c > 7):
    #             result[2][2] = 1
    #             break
    #         result[r][c] = coefficentMatrix[r * 3 + c]
    return transformation_matrix


def trasnform_image(image, T):
    #Get width and height.
    (image_width, image_height) = image.shape

    #Initialize new image
    new_image = np.zeros(image.shape)
    
    # create a matrix representing the indices of columns in the first row
    col_indices = np.arange(image_width)

    # create an array representing the row index
    row_index = np.zeros(image_width)

    # create an array for the homogenoues coordinate
    third = np.ones(image_width)
    
    # construct a matrix 3xWidth
    row_coordinates = np.array([row_index, col_indices, third]).transpose()

    #Multiply by transformation to get the new coordinates
    new_coordinates = np.around(np.matmul(row_coordinates, T))
    
    # for row_number in range(1, image_height):

    #     # create a vector with the current row number
    #     row_index = np.ones(image_width) * row_number

    #     # update the matrix for the current row
    #     row_coordinates = np.array([row_index, col_indices, third]).transpose()
        
    #      #Multiply by transformation to get the new coordinates
    #     new_coordinates = np.around(np.matmul(row_coordinates, T))

    #     # update new image based on the transformation
    #     for col_number in range(0,image_width):

    #         # Try using the regular transfomation instead of the inverse:

    
    r = 0
    c = 0

     

    xMax = 0
    yMax = 0
    xMin = 0 
    yMin = 0 
    count =0
    tInverted = np.linalg.inv(T)
    print(T)
    print(tInverted)

    for r in range(0,image_width):
        for c in range(0, image_height):
            #print(f"{r} - {c}")
            sourceVector = np.matmul(tInverted,np.array([r, c, 1 ]).transpose())
            # sourceVector = np.matmul(T,np.array([r, c, 1 ]).transpose())
            #print(f"{sourceVector[2][0][0]}")
            normSourceVector = (sourceVector / sourceVector[2])
            x = normSourceVector.item(0)
            y = normSourceVector.item(1)
            xSrc = int(round(x))
            ySrc = int(round(y))
            
            if(x > xMax):
                xMax = x

            if ySrc >= image_height or ySrc < 0:
                new_image[r][c] = 0
                #ySrc = c
            elif xSrc >= image_width or ySrc < 0:
                new_image[r][c] = 0
                #xSrc = r
            else:
                new_image[r][c] = image[xSrc][ySrc]
                count+=1
    return new_image


def create_wormhole(im, T, iter=5):
    (image_width, image_height) = im.shape
    new_image = trasnform_image(im, T)
    return new_image

def get_psuedo_invert(target_points, source_points):
    
    # initialize transformation matrix. 
    t = np.zeros((target_points.size, 8))

    # initialize row counter
    r = 0

    # build two rows per point. 
    for sPoint,tPoint in zip(source_points,target_points):

        x = sPoint[0]
        y = sPoint[1]
        xTag = tPoint[0]
        yTag = tPoint[1]

        # build first row
        t[2 * r] = np.array([x, y, 0, 0, 1, 0, -x * xTag, -y * xTag])

        # build second row
        t[2 * r + 1] = np.array([0, 0, x, y, 0, 1, -x *yTag, -y * yTag])
        r += 1
    
    mul = np.matmul(t.transpose(),  t)
    invers = np.linalg.inv(mul)
    pinv = np.matmul( invers , t.transpose())


    return pinv