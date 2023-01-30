"""
Bioinformatics Algorithms - Project I

W tym pliku przeprowadzone zostały testy modułu "matrix_operations".
"""

import matrix_operations as m

#tworzymy naszą macierz - w tym wypadku jest to macierz zbudowana z list w listach
m1 = [[1,2,3], [1,5,4], [0,2,1]]

print("\nJest to macierz", len(m1), "x", len(m1[0]))
print("Prezentacja macierzy m1:\n", m1)
print("")

print("Suma wszystkich elementow macierzy m1:", m.sum_matrix(m1))
print("Najwieksza wartosc elementow macierzy m1:", m.largest_value(m1))
print("Najmniejsza wartosc elementow macierzy m1:", m.smallest_value(m1))
print("Srednia wartosc elementow macierzy m1:", m.mean_value(m1))
print("Srednie wartosci w wierszach w postaci listy:", m.mean_row(m1))
print("Pomnozone wartosci po przekatnej:", m.diagonal_multiplication(m1))
print("Sprawdzenie czy liczba kolumn jest taka sama jak liczba wierszy:", m.checking_square_matrix(m1))
print("\nMacierz z pomnożonymi wartosciami przez wybrana liczbe:\n", m.values_multiplication(m1))

#tworzymy dwie dodatkowe macierze do testow funkcji o tych samych wymiarach
m2 = [[2,0,1], [4,2,3], [2,3,2]]
m3 = [[1,2,3], [2,0,2], [1,0,4]]

print("\n\nMacierz m2:", m2)
print("Macierz m3:", m3)
print("")

print("Macierze m1 i m2 maja takie same wymiary(3x3):", m.checking_new_matrix(m1,m2))
print("Macierze m1 i m3 maja takie same wymiary(3x3):", m.checking_new_matrix(m1,m3))
print("\nWynik mnozenia macierzy m2 i m3:\n", m.matrix_multiplication(m2,m3))

