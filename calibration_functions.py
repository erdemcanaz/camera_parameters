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

def least_squares_solver(array_A, array_B):
    """
    This function solves the equation Ax = B using least squares method.

    INPUT: 
    array_A - 2D np array of size m x n
    array_B - 2D np array of size m x 1

    OUTPUT: 
    array_X - 1D list of size n x 1

    """

    array_X, _, _, _ = np.linalg.lstsq(array_A, array_B, rcond=None)  # "rcond = None" means numbers smaller than machine precision are considered zero
    array_X = array_X.flatten() # Convert to a 1D list
    array_X = array_X.tolist()  # Convert to a 1D list
    return array_X

if __name__ == "__main__":
    # The cost of the apple and carrot  example given in the presentation to be solved using least squares method
    
    # 1)Ayse says that she has bought two apples and three carrot for 3.49₺.
    #  *It is known that at that time apple and carrot cost 0.47₺ and 0.85₺ per item respectively.
    # 2)Ahmet says that she has bought seven apples and five carrot for 7.60₺.
    #  *It is known that at that time apple and carrot cost 0.55₺ and 0.75₺ per item respectively.
    # 3)Ezgi says that she has bought two apples and four carrot for 4.20₺.
    #  *It is known that at that time apple and carrot cost 0.50₺ and 0.80₺ per item respectively.

    # Find the cost of an apple and a carrot.
    array_A = np.zeros((3, 2))
    array_B = np.zeros((3, 1))

    array_A[0] = [2, 3]
    array_A[1] = [7, 5]
    array_A[2] = [2, 4]

    array_B[0] = [3.49]
    array_B[1] = [7.60]
    array_B[2] = [4.20]

    apple_cost, carrot_cost = least_squares_solver(array_A, array_B)
    print('Apple cost: {:.2f} TL'.format(apple_cost))
    print('Carrot cost: {:.2f} TL'.format(carrot_cost))

    # when run, the output is:
    #Apple cost: 0.51 TL
    #Carrot cost: 0.80 TL
