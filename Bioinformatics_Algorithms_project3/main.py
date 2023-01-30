"""
Bioinformatics Algorithms - Project III

moduł testowy pliku wunsch_functions
"""

from wunsch_functions import create_submat, Needleman_Wunsch_ties_version, recover_alignment_ties_version

#stworzenie slownika ze score wszystkich zlepkow
scoring = create_submat(1,-1, 0, '-ACTG')

score, traceback = Needleman_Wunsch_ties_version('CGATA', 'CAGTA', scoring)
alignments = recover_alignment_ties_version(traceback, 'CGATA', 'CAGTA')

print("CGATA x CAGTA")
print('Score:')
for i in range(0, len(score)):
        print(score[i])
        
#wydruk traceback razem z remisami
print("\nTraceback:")
for i in range(0, len(traceback)):
        print(traceback[i])

print("\nOptymalne powiazania dla CGATA x CAGTA:")
print(alignments)


print("\n\nOptymalne powiazania dla ATGCAA x AGTCTA:")
scoring2 = create_submat(3,-3, 0, '-ACTG')
traceback2 = Needleman_Wunsch_ties_version('ATGCAA', 'AGTCTA', scoring2)[1]
al = recover_alignment_ties_version(traceback2, 'ATGCAA', 'AGTCTA')
print("Ilosc optymalnych powiazan:", len(al))
print(al)


traceback3 = Needleman_Wunsch_ties_version('CATTACT', 'CGGTATC', scoring2)[1]
al2= recover_alignment_ties_version(traceback3, 'CATTACT', 'CGGTATC')
print("\n\nOptymalne powiazania dla CATTACT x CGGTATC")
print("Ilosc optymalnych powiazan:", len(al2))
print(al2)

