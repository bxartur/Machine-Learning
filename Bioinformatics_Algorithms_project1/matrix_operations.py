"""
Bioinformatics Algorithms - Project I

Tresc polecenia:
chapter 2, ex2. 
Write a module in Python including a set of functions working over matrices. The matrix
(represented as a list of lists) will be passed as the first argument. Some functions to
include may be the following:
    a. calculate the sum of the values in the matrix;
    b. indicate the largest (smallest) value in the matrix;
    c. calculate the mean of the values in the matrix;
    d. calculate the mean (or sum) of the values in each row (or column); the result should be a list;
    e. calculate the multiplication of the values in the diagonal;
    f. check if the matrix is square (same number of rows and columns);
    g. multiply all elements by a numerical value, returning another matrix;
    h. add two matrices (assuming they have the same dimension);
    i. multiply two matrices.
"""

#podpunkt a - suma wszystkich elementow macierzy
def sum_matrix(matrix):
    s=[]
    i,j = 0,0
    for i in matrix:
        for j in i:
            s.append(j)
    return sum(s)

assert sum_matrix([[1,2,3], [1,2,3], [1,2,3]]) == 18
assert sum_matrix([[1,2], [2,1]]) == 6


#podpunkt b1 - najwieksza wartosc w macierzy(maximum)
def largest_value(matrix):
    l=[]
    i,j = 0,0
    for i in matrix:
        for j in i:
            l.append(j)
    return max(l)

assert largest_value([[3,2,2], [7,2,3], [1,2,1]]) == 7
assert largest_value([[5,2], [3,1]]) == 5


#podpunkt b2 - najmniejsza wartosc w macierzy(minimum)
def smallest_value(matrix):
    s=[]
    i,j = 0,0
    for i in matrix:
        for j in i:
            s.append(j)
    return min(s)

assert smallest_value([[0,3], [2,1]]) == 0
assert smallest_value([[11,2,2], [7,4,3], [1,2,2]]) == 1


#podpunkt c - srednia wartosc wszystkich elementow (suma / ilosc wszystkich elementow)
def mean_value(matrix):
    m=[]
    i,j = 0,0
    for i in matrix:
        for j in i:
            m.append(j)
    return sum(m) / (len(m))

assert mean_value([[1,3], [2,2]]) == 2
assert mean_value([[2,3,4], [1,3,5], [2,3,4]]) == 3


#podpunkt d - srednia wartosc kazdego wiersza w postaci listy
def mean_row(matrix):
    r=[]
    i,j = 0,0
    for i in matrix:
        suma,mean = 0,0
        for j in i:
            suma = suma + j
            mean = suma / (len(matrix))
        r.append(mean)
    return r

assert mean_row([[1,3], [5,7]]) == [2,6]
assert mean_row([[1,2,3], [4,5,6], [7,8,9]]) == [2,5,8]


#podpunkt e - mnozenie wartosci po przekatnej
def diagonal_multiplication(matrix):
    m=1 #mnoznik(wartosc przez jaka chcemy przemnozyc przekatna)
    i=0
    for i in range(len(matrix)):
        m = m * matrix[i][i]
    return m

assert diagonal_multiplication([[2,1], [3,4]]) == 8
assert diagonal_multiplication([[2,1,1], [3,2,3], [1,2,3]]) == 12


#podpunkt f - sprawdzenie czy macierz jest kwadratowa (czy ma tyle samo kolumn co wierszy)
def checking_square_matrix(matrix):
    i=0
    for i in matrix:
        if len(i) != len(matrix):
            return False
    return True

assert checking_square_matrix([[1,1], [1,1]]) == True
assert checking_square_matrix([[1,1,1], [1,1], [1,1,1]]) == False


#podpunkt g - stworzenie nowej macierzy z przemnozonymi (przez wybrana liczbe) elementami z poprzedniej
def values_multiplication(matrix):
    count = 2 #mnoznik(wybrana przez nas wartosc)
    multiplicated = []
    for i in range(len(matrix)):
        multiplicated.append(list())
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            multiplicated[i].append(matrix[i][j] * count)
    return multiplicated

assert values_multiplication([[1,2], [3,4]]) == [[2,4], [6,8]]
assert values_multiplication([[1,1,1], [2,2,2], [3,3,3]]) == [[2,2,2], [4,4,4], [6,6,6]]


#podpunkt h - sprawdzenie czy dwie wybrane macierze maja takie same wymiary
def checking_new_matrix(m1, m2):
    if len(m1) != len(m2):
        return False
    if len(m1[0]) != len(m2[0]):
        return False
    else:
        return True

assert checking_new_matrix([[1,1],[2,2]], [[1,2],[2,1]]) == True
assert checking_new_matrix([[1,1,1],[2,2,2],[3,3,3]], [[1,1],[2,2]]) == False


#podpunkt i - mnozenie dwoch macierzy
def matrix_multiplication(m1, m2):
    multiplicated = []
    for i in range(len(m1)):
        multiplicated.append(list())
    i,j,k = 0,0,0
    for i in range(len(m1)):
        for j in range(len(m2[0])):
            x=0
            for k in range(len(m1[0])):
                x = x + m1[i][k] * m2[k][j]
            multiplicated[i].append(x)
    return multiplicated

#przy tworzeniu assertow do funkcji mnozacej dwie macierze obliczenia wykonalem na kartce, zeby sprawdzic poprawnosc dzialania
assert matrix_multiplication([[1,2,0],[2,2,2],[2,1,1]], [[1,2,1],[0,0,1],[1,0,2]]) == [[1,2,3], [4,4,8], [3,4,5]]
assert matrix_multiplication([[2,2],[1,1]], [[2,1],[1,2]]) == [[6,6], [3,3]]




