"""
Bioinformatics Algorithm - Project II

chapter 3, ex 6
Files from UniProt saved in the FASTA format have a specific header structure given by:
   
    db|Id|Entry Protein OS = Organism [GN = Gene] PE = Existence SV = Version

Write a function that using regular expressions parses a string in this format and returns
a dictionary with the different fields (the key should be the field name). Note the part in
right brackets is optional, the parts in italics are the values of the fields, while the parts in
upper case are constant placeholders.
"""

import re

def uniprot_header_dictionary(uniprot_header):    
        
    #przeszukiwanie naglowka w celu otrzymania wartosci dla db, id, entry, protein i os
    structure_1= '>(.*)\|(.*)\|(\S*) (.*) OS=(.*[)])'
    fill_1= re.search(structure_1, uniprot_header)
        
    #sprawdzenie czy doszlo do wypelnienia pierwszej czesci
    if fill_1 == None:
        return print("uncorrect uniprot header")
    
    #sprawdzenie czy w podanym naglowku wystepuje GN i ew. zapis jego wartosci
    if uniprot_header.find('GN=') > -1:
        structure_gn= 'GN=(.*) P'
        gn= (re.search(structure_gn, uniprot_header)).group(1)
    else:
        gn= None
    
    #kontynuacja przeszukiwania w celu otrzymania wartosci dla pe i sv
    structure_2= 'PE=(.*) SV=(.*)'
    fill_2= re.search(structure_2, uniprot_header)
    
    if fill_2 == None:
        return print("uncorrect uniprot header")
    
    
    #tworzymy slownik w oparciu o schemat naglowka:
    #db|Id|Entry Protein OS=Organism [GN=Gene] PE=Existence SV=Version
    header_content = {
        'db': fill_1.group(1),
        'id': fill_1.group(2),
        'entry': fill_1.group(3),
        'protein': fill_1.group(4),
        'OS': fill_1.group(5),
        'GN': gn,
        'PE': fill_2.group(1),
        'SV': fill_2.group(2)
        }
    
    return header_content


#przykladowe naglowki sprawdzajacy dzialanie funkcji
assert uniprot_header_dictionary(">a|b|c d OS=e (f) GN=g PE=1 SV=1") == {'db':'a','id':'b','entry':'c','protein':'d','OS':'e (f)','GN':'g','PE':'1','SV':'1'}
assert uniprot_header_dictionary(">a|b|c d OS=e (f) PE=1 SV=1") == {'db':'a','id':'b','entry':'c','protein':'d','OS':'e (f)','GN':None,'PE':'1','SV':'1'}