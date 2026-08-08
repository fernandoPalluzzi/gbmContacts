# gbmContacts
The **gbmContacts** software characterizes and draws atigen-antibody interaction profiles based on a simple input sequence of interactions between antibody (Ab) and antigen (Ag) and an input contact library.
It is a lightweight version of [lstmContacts](https://github.com/fernandoPalluzzi/lstmContacts), providing a fast and accurate solution to predict a new antigen's variant impact, by searching and evaluating the affinity of alibrary of known antibodies to a given antigen.

What **gbmContacts** can do:
- Generate a simulated Ab-Ag interaction profile and predict if a complex will be stable over time.
- Evaluate the affinity of every interaction in an Ab-Ag complex, providing stability and confidence metrics.
- Assign risk score and risk profiling to the Ab-Ag complex with no need for computationally-intensive molecular dynamics simulations.
- Fast searching for the best neutralizing antobody across the (customizable) internal library.
- Compiling of a technical PDF report of the Ab-Ag complex evaluation (preliminary version).

What **gbmContacts** cannot do:
- Replace experimentally proven interaction Ab-Ag profiling.
- Replace direct biomedical evidence or expert's knowledge.

## Installation

Just download or clone this library, navigate to the library root and run:

```
pip install -e .
```

#### Requirements

The current version of **gbmContacts** (v1.0.0) requires **python** (>=3.8), **pandas** (>=3.0.2), **numpy** (>=2.4.4), **scipy** (>=1.17.1), **scikit-learn** (>=1.8.0), **matplotlib** (>=3.10.8), **seaborn** (>=0.13.2), and **lightgbm** (>=4.6.0).

PDF report generation requires **reportlab** (>=5.0.0) installation, but it is not mandatory and all other gbmContact functionalities will be available in any case.

&nbsp;

## 1. Internal representation of an antigen-antibody complex

The input **antigen-antibody complex** (AAC) is represented by a list x = [x1, x2, …, xn] of n **contacts**. Each j-th contact is a vector x_j = [x_j0, x_j1, …, x_jm] of m elements, where x_j0 is the antibody residue interacting with the x_j1, …, x_jm residues on the surface of the receptor binding domain (RBD) of the Spike protein variant. Every contact corresponds univocally to a vector ai = affinity(x_j) of 101 affinity score values, ranging in affinity value from 0 to 1, and corresponding to the 101 nanoseconds of the molecular dynamics simulation stored in the internal library (`gbmContacts/data/Affinity_data.txt`). In the internal library lookup table (see the next sections), antibodies are reported with the corresponding protein data bank ([**PDB**](https://www.rcsb.org/)) 3D structure ID: 7kmg (Bamlanivimab, Ly-Cov555), 7c01 (Etesevimab, Ly-Cov016), 7cm4 (Regdanvimab, CTP-59), 7l7d (Tixagevimab, AZD8895), 7l7e (Cilgavimab, AZD1061), 7r6w (Sotrovimab), 6zcz (EY6A). The COVID-19 variants for which a molecular dynamic simulation is available in the built-in library include: *wt* (wild-type), *alpha*, *beta*, *delta*, *omicron*. The input AAC wil be searched, both exactly and by similarity, against these data.

The high-level user-interfacing structure for an AAC is a plain Python dictionary of contacts:

```py
x = {'contact_1': ['h.R50', 'V483', 'E484'],
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
