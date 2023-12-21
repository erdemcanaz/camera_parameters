import numpy as np

def calculate_projection_matrix_using_test_data(test_data = None):
    """
    This function calculates the projection matrix using test data by utilizing least squares method.

    INPUT: test_data - list of atleast 6 lists of 5 elements    
    test_data = [ [u1, v1, x1, y1, z1], [u2, v2, x2, y2, z2], ..., [un, vn, xn, yn, zn] ]
    where u, v are pixel coordinates and x, y, z are world coordinates

    OUTPUT: 
    projection_matrix = [ p11, p12, p13, p14, p21, p22, p23, p24, p31, p32, p33, p34 ]

    """

    number_of_test_data = len(test_data)

    if number_of_test_data < 6:
        #Each test data results in 2 equations, so atleast 6 test data is required to solve for 12 unknowns
        raise Exception('Atleast 6 test data point is required')

    array_A = np.zeros(shape=(number_of_test_data*2, 12)) 
    array_B = np.zeros(shape=(number_of_test_data*2, 1)) # zero vector

    for test_data_no, single_test_data in enumerate(test_data):                
        if len(single_test_data) != 5:
            raise Exception('Each test data should have 5 elements')            
        for element in single_test_data:
            if not isinstance(element, (int, float)): #number related types of python
                raise Exception('Each element of test data should be a number')
                    
        u, v, x, y, z = single_test_data
        u_related_row = [  x, y, z, 1,  0,  0,  0,  0, -u*x, -u*y, -u*z, -u ]
        v_related_row = [  0, 0, 0, 0,  x,  y,  z,  1, -v*x, -v*y, -v*z, -v ]

        array_A[2*test_data_no] = u_related_row
        array_A[2*test_data_no+1] = v_related_row

    # Apply least squares method
    projection_matrix, _, _, _ = np.linalg.lstsq(array_A, array_B, rcond=None)  # "rcond = None" means numbers smaller than machine precision are considered zero
    np_projection_matrix = projection_matrix.flatten() # Convert to a 1D list
    projection_matrix = np_projection_matrix.tolist()  # Convert to a 1D list

    return projection_matrix


calculate_projection_matrix_using_test_data(test_data = [[1,2,3,4,5],[6,7,8,9,10], [11,12,13,14,15], [16,17,18,19,20], [21,22,23,24,25], [26,27,28,29,30]])