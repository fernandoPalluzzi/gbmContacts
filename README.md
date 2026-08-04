# lstmContacts
The **gbmContacts** software characterizes and draws atigen-antibody interaction profiles based on a simple input sequence of interactions between antibody (Ab) and antigen (Ag) and an input contact library.
It is a lightweight, fast, and accurate solution to predict a new antigen's variant impact, by searching and evaluating the efficacy of known antibodies, without requiring computationally-intensive molecular dynamics simulations

What **gbmContacts** can do:
- Generate a simulated Ab-Ag interaction profile and predict if a complex will be stable over time.
- Evaluate the stability of every interaction in an Ab-Ag complex.
- Assign risk score and risk profiling to the Ab-Ag complex.
- Evaluate the confidence of complex stability with antibodies included the available contact library.
- Compiling a technical PDF report of the Ab-Ag complex evaluation (preliminary version).

What **gbmContacts** cannot do:
- Replace experimentally proven interaction Ab-Ag profiling.
- Replace direct biomedical evidence or expert's knowledge.

## Requirements

This is a small Python library, with a small set of requirements:
- 

## Installation

Just download or clone this library, navigate to the library root and run:

```
pip install -e .
```

## 1. Internal representation of an antigen-antibody complex

The input **antigen-antibody complex** (AAC) is represented by a list x = [x1, x2, …, xn] of n **contacts**. Each j-th contact is a vector xj = [x0j, x1j, …, xmj] of mj elements, where x0j is the antibody residue interacting with the x1j, …, xmj residues on the surface of the receptor binding domain (RBD) of the Spike protein variant. Every contact corresponds univocally to a vector ai = affinity(xj) of 101 affinity score values, ranging from 0 to 1, and corresponding to the 101 nanoseconds of the molecular dynamics simulation stored in the internal library (object `contact.data`). In the internal library, antibodies are reported with the corresponding protein data bank ([**PDB**](https://www.rcsb.org/)) 3D structure ID: 7kmg (Bamlanivimab, Ly-Cov555), 7c01 (Etesevimab, Ly-Cov016), 7cm4 (Regdanvimab, CTP-59), 7l7d (Tixagevimab, AZD8895), 7l7e (Cilgavimab, AZD1061), 7r6w (Sotrovimab), 6zcz (EY6A). The variants for which a molecular dynamic simulation is available in the built-in library include: *wt* (wild-type), *alpha*, *beta*, *delta*, *omicron*. The input AAC wil be searched, both exactly and by similarity, against these data.

The input AAC must be given as a dictionary of interactions:

```py
{'contact_1': ['h.R50', 'V483', 'E484'],
 'contact_2': ['h.L55', 'L452', 'T470', 'F490'],
 'contact_3': ['h.Y101', 'E484', 'F490'],
 'contact_4': ['h.R104', 'Q493', 'S494'],
 'contact_5': ['l.Y32', 'F486', 'Y489'],
 'contact_6': ['l.Y92', 'F486', 'Y489'],
 'contact_7': ['l.R96', 'V483', 'E484']}
```

In the example above, `x` is the AAC and each vector in `x` is a contact. The first element of each contact is always the amino acid residue of the antibody, defined by the FAB chain ("h" for "heavy" and "l" for "light"), followed by a dot, the single letter code of the residue, and its position in the polypeptide chain. The other elements of a contact are the antigen residues interacting with the antibody one.

## 2. Internal data structures
...
