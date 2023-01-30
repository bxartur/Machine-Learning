"""
Bioinformatics Algorithm - Project II

moduł testowy pliku projekt3.py
"""

from projekt3 import uniprot_header_dictionary

#przykladowy naglowek pliku fasta z uniprot
header1 = ">sp|Q9KM66|CQSS_VIBCH CAI-1 autoinducer sensor kinase/phosphatase CqsS OS=Vibrio cholerae serotype O1 (strain ATCC 39315 / El Tor Inaba N16961) GN=cqsS PE=1 SV=1"

print("\nSlownik z zawartoscia naglowka pliku FASTA (przyklad 1):",)
print(uniprot_header_dictionary(header1))

header2 = ">sp|P39735|SAW1_YEAST Single-strand annealing weakened protein 1 OS=Saccharomyces cerevisiae (strain ATCC 204508 / S288c) PE=1 SV=1"

print("\nSlownik z zawartoscia naglowka pliku FASTA (przyklad 2):",)
print(uniprot_header_dictionary(header2))