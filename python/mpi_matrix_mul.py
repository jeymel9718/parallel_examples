"""
Sample code of matrix multiplication using MPI
"""

from mpi4py import MPI
from random import randint
import time

N=500

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
workers = comm.Get_size() - 1

def init_matrix(n):
    """
    Initialize matrix NxN with random values
    """
    X = [[randint(0, 9) for i in range(N)] for j in range(n)]
    Y = [[randint(0, 9) for i in range(N)] for j in range(n)]
    return X, Y

def matrix_mul(X, Y):
    """
    Execute the matrix mul operation
    """
    Z = []
    for X_row in X:
        Z_row = []
        for Y_col in zip(*Y):
            Z_row.append(sum(a*b for a, b in zip(X_row, Y_col)))
        Z.append(Z_row)
    
    return Z

def matrix_distribuite(X, Y):
    """
    Distribute rows of first matrix and whole second matrix to workers, this
    done via master node. Then workers calculate a sub matrix by multiplying
    incoming matrix and send result back to master
    """
    def split_matrix(seq, p):
        """
        Split matrix into small parts according to the no of workers. These
        parts will be send to slaves by master node
        """
        rows = []
        n = len(seq) / p
        r = len(seq) % p
        b, e = 0, n + min(1, r)
        for i in range(p):
            end = int(e)
            rows.append(seq[b:end])
            r = max(0, r - 1)
            b, e = end, e + n + min(1, r)

        return rows

    rows = split_matrix(X, workers)

    pid = 1
    for row in rows:
        comm.send(row, dest=pid, tag=1)
        comm.send(Y, dest=pid, tag=2)
        pid = pid + 1

def matrix_merge():
    """
    Assemble returning values form salves and generate final matrix. Slaves
    calculate single rows of final matrix
    """

    Z = []
    pid = 1
    for n in range(workers):
        row = comm.recv(source=pid, tag=pid)
        Z = Z + row
        pid = pid + 1
    
    return Z


if __name__ == '__main__':
    """
    Main method:
        - Initialize a dummy matrix
        - The master will split the matrix and send it to the workers
        - The workers will recive the matrix data and calculate the multiplication
        - The master will recive each data from the workers and merge it
    """
    # Master node code
    if rank == 0:
        X, Y = init_matrix(N)
        
        # start time
        start = time.time()
        
        # Distribute the data amoung the workers
        matrix_distribuite(X, Y)

        # Merge the matrix recived from the workers
        result = matrix_merge()
        
        # end time
        end = time.time()

        print("Elapsed time: ", end-start)
    # Workers code
    else:
        # Obtain the data from the master
        x = comm.recv(source=0, tag=1)
        y = comm.recv(source=0, tag=2)
        # Execute the matrix multiplication
        z = matrix_mul(x, y)
        # Send data back to master
        comm.send(z, dest=0, tag=rank)