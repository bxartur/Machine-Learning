"""
Bioinformatics Algorithms - Project III

chapter 6, ex 4/161 
    Consider the functions to calculate pairwise global alignments. Note that, in the case
there are distinct alignments with the same optimal score, the functions only return one
of them. Notice that these ties arise in the cases where, in the recurrence relation of the
DP algorithm, there are at least two alternatives that return the same score.

a. Define a function needleman_Wunsch_with_ties, which is able to return a traceback
matrix (T) with each cell being a list of optimal alternatives and not a single
one.

b. Define a function recover_align_with_ties, which taking a trace-back matrix created
by the previous function, can return a list with the multiple optimal alignments.
"""

#utworzenie slownika ze score wszystkich elementow naszego alfabetu
def create_submat(match, mismatch, gap, alphabet):
    sm = {}
    for c1 in alphabet:
        for c2 in alphabet:
            if c1 == '-' or c2 == '-':
                sm[c1+c2] = gap
            elif c1 == c2:
                sm[c1+c2] = match
            else:
                sm[c1+c2] = mismatch
    return sm

assert create_submat(3, -1, -3, '01') == {'00':3, '01':-1, '10':-1, '11':3}
assert create_submat(1, -1, 0, 'A') == {'AA': 1}


#podpunkt a - zdefiniowanie funkcji needlemana-wunscha z remisami
def max3t_ties_version(v1, v2, v3):
    if v1 > v2:
        if v1 > v3: return [1]
        elif v3 > v1: return [3]
        else: return [1, 3] #remis v1==v3
    elif v2 > v1:
        if v2 > v3: return [2]
        elif v3 > v2: return [3]
        else: return [2, 3] #remis v2==v3
    else: #remis v1 == v2
        if v1 > v3: return [1, 2]
        elif v3 > v1: return [3]
        else: return [1, 2, 3] #remis v1==v3

assert max3t_ties_version(1, 1, 2) == [3]
assert max3t_ties_version(2, 1, 2) == [1,3]
assert max3t_ties_version(1, 2, 2) == [2,3]
assert max3t_ties_version(1, 1, 1) == [1,2,3]


def Needleman_Wunsch_ties_version(seq1, seq2, scoring):
    S = [[0]]
    T = [[0]]
    gap_score = scoring['--']

    #tworzenie rzędu gaps
    for j in range(1, len(seq2)+1):
        S[0].append(gap_score * j)
        T[0].append(3)
    
    #tworzenie kolumny gaps
    for i in range(1, len(seq1)+1):
        S.append([gap_score * i])
        T.append([2])
   
    #wypielniamy reszte macierzy
    for i in range(0, len (seq1)):
        for j in range(len(seq2)):
            s1 = S[i][j] + scoring[seq1[i] + seq2[j]]
            s2 = S[i][j+1] + gap_score
            s3 = S[i+1][j] + gap_score
            S[i+1].append(max(s1, s2, s3))
            T[i+1].append(max3t_ties_version(s1, s2, s3))
    
    return(S, T)


sc = create_submat(3, -1, -3, '01-')
assert Needleman_Wunsch_ties_version('0', '0', sc)[0] == ([[0, -3], [-3, 3]]) #score
assert Needleman_Wunsch_ties_version('0', '0', sc)[1] == ([[0, 3], [2,[1]]]) #traceback
assert Needleman_Wunsch_ties_version('10', '01', sc)[0] == ([[0,-3,-6], [-3,-1, 0], [-6, 0,-2]]) #score
assert Needleman_Wunsch_ties_version('10', '01', sc)[1] == ([[0, 3, 3], [2,[1],[1]], [2,[1],[1]]]) #traceback


#podpunkt b - zdefiniowanie funkcji recover_alignment z remisami
def recover_alignment_ties_version(traceback, seq_1, seq_2):
    result = []
    #rozpoczynamy w prawym dolnym rogu(od konca)
    alignments = [['', '', len(seq_1), len(seq_2)]]
    
    while alignments:
        alignment= alignments.pop(0)
        i, j = alignment[2], alignment[3]
        
        if i>0 or j>0:
            #jako ze traceback[i][j] przez remisy przyjmuje czasami wiecej niz
            #jedna wartosc przydatne jest stworzenie petli for
            value = 0
            for value in traceback[i][j]:
                alignment_list = []
                if value == 1: 
                    #zawartosc dodajemy zgodnie z ukladem listy w wierszu 79, czyli seq_1, seq_2, i, j
                    alignment_list.append(seq_1[i-1] + alignment[0])
                    alignment_list.append(seq_2[j-1] + alignment[1])
                    alignment_list.append(i-1)
                    alignment_list.append(j-1)
               
                elif value == 2:
                    alignment_list.append(seq_1[i-1] + alignment[0])
                    alignment_list.append("-" + alignment[1])
                    alignment_list.append(i-1)
                    alignment_list.append(j)
                
                elif value == 3:
                    alignment_list.append('-' + alignment[0])
                    alignment_list.append(seq_2[j-1] + alignment[1])
                    alignment_list.append(i)
                    alignment_list.append(j-1)
                alignments.append(alignment_list)
        elif i == 0 and j == 0:
            result.append(alignment[:2])
    
    return result


#dodatkowe testy
sc2 = create_submat(2,-2, 0, '-ACTG')
trac = Needleman_Wunsch_ties_version('TAG', 'TAA', sc2)[1]
assert recover_alignment_ties_version(trac, 'TAG', 'TAA') == [['T-AG','TAA-'],['TA-G','TAA-'],['TAG-','TA-A']]

trac2 = Needleman_Wunsch_ties_version('CAGT', 'CACT', sc2)[1]
assert recover_alignment_ties_version(trac2, 'CAGT', 'CACT') == [['CA-GT','CAC-T'],['CAG-T','CA-CT']]

trac3 = Needleman_Wunsch_ties_version('TAGAT', 'TAATT', sc2)[1]
assert recover_alignment_ties_version(trac3, 'TAGAT', 'TAATT') == [['TAGA-T','TA-ATT'], ['TAGAT-','TA-ATT']]

trac4 = Needleman_Wunsch_ties_version('TACGTA', 'TAGCTA', sc2)[1]
assert recover_alignment_ties_version(trac4, 'TACGTA', 'TAGCTA') == [['TA-CGTA','TAGC-TA'],['TACG-TA','TA-GCTA']]






