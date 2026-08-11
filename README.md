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

The input **antigen-antibody complex** (AAC) is represented by a list $x = [x_1, x_2, …, x_n]$ of n **contacts**. Each $j$-th contact is a vector $x_j = [x_{j0}, x_{j1}, …, x_{jm}]$ of $m$ elements, where $x_{j0}$ is the antibody residue interacting with the $x_{j1}, …, x_{jm}$ residues on the surface of the receptor binding domain (RBD) of the Spike protein variant. Every contact corresponds univocally to a vector $a_i = \mathrm{affinity}(x_j)$ of 101 affinity score values, ranging in affinity value from 0 to 1, and corresponding to the 101 nanoseconds of the molecular dynamics simulation stored in the internal library (`gbmContacts/data/Affinity_data.txt`). In the internal library lookup table (see the next sections), antibodies are reported with the corresponding protein data bank ([**PDB**](https://www.rcsb.org/)) 3D structure ID: 7kmg (Bamlanivimab, Ly-Cov555), 7c01 (Etesevimab, Ly-Cov016), 7cm4 (Regdanvimab, CTP-59), 7l7d (Tixagevimab, AZD8895), 7l7e (Cilgavimab, AZD1061), 7r6w (Sotrovimab), 6zcz (EY6A). The COVID-19 variants for which a molecular dynamic simulation is available in the built-in library include: *wt* (wild-type), *alpha*, *beta*, *delta*, *omicron*. The input AAC wil be searched, both exactly and by similarity, against these data.

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

All data structures can be saved locally as tab-separated text file. The default dataset is stored in `gbmContacts/data`. 

#### 2.1. Affinity data and metadata

All AAC predictions depend on a set of internal data structures. The two most important ones are the **affinity data table** and the **affinity metadata**. The former is a table of Ab residues, encoded in the form CHAIN.ResiduePosition, where CHAIN can be either light (l) or heavy (h), Residue is the single-letter code of one of the known amino acids, and Position is the position of the residue in the polypeptide chain. For instance, l.R50 is an Arginine residue, the 50th of the light Ab chain. The number of rows of the affinity table corresponds to the total length of the measured Ab-Ag interaction time frame. The current default COVID-19 library has 101 measurement: one per nanosecond. Column names can be repeated, since every column refers to the interaction between the Ab residue with a corresponding Ag residue. Here below we have an example of 9 Ab residues and 6 affinity measures:

```
h.R50  h.L55  h.Y101 h.R104 l.Y32  l.Y92  l.R96  h.R50  h.L55
1      1      1      1      1      1      1      1      1
0.99   1      1      0.86   0.98   1      1      0.99   0.98
1      0.96   1      0.93   1      1      1      1      0.9
1      0.89   1      0.74   0.96   1      1      1      0.97
1      0.87   1      0.81   0.88   1      0.99   0.99   0.98
1      0.95   1      0.97   0.87   1      0.99   1      0.92
```

The affinity metadata is instead a table with 3 attributes:
- **antibody**: a single-word Ab name (e.g., the PDB ID);
- **variants**: a single word Ag name, where "wt" is reserved to the wild-type variant;
- **repricates**: an integer indicating the replicate number (a minimum of 3 replicates is recommended).

The number of rows of the metadata table must have the same number of columns of the affinity table, to map the antibody name, varant, and replicate to the affinity data table.

#### 2.2. Contact map

The contact map is a table reporting every single known Ab-Ag interaction per row, described by 3 attributes:
- **antibody**: a single-word Ab name (e.g., the PDB ID) from the same set of names used in the metadata;
- **residue**: the Ab residue participating in the Ab-Ag interaction, must follow the same format as the affinity data table (i.e., <CHAIN>.<RESIDUE><POSITION>).
- **contact**: the Ag residue participating in the Ab-Ag interaction, must follow the format <RESIDUE><POSITION>.

#### 2.3. Contact mutations

The contact mutations table reports a known mutation per row, and includes the following attributes:
- **variant**: a variant name consistent with the ones used in the metadata table;
- **wt**: wild-type Ag residue in the form ResiduePosition (e.g., N501);
- **mutant**: mutant Ag residue in the form ResiduePosition (e.g., Y501);
- **wt.group**: wt amino acid biochemical group (see next section);
- **mutant.group**: mutant amino acid biochemical group (see next section).

#### 2.4. Amino acid biochemical properties

This table reports the amino acid name (aa), three-letters code (tlc), single-letter code (code), biochemical group (group), and a free-text description (description), for each known amino acid.
Attributes aa, code, and group are used to evaluate the predicted affinity.

```
            aa tlc code     group                         description
       Alanine Ala    A aliphatic                    Nonpolar neutral
    Isoleucine Ile    I aliphatic                    Nonpolar neutral
       Leucine Leu    L aliphatic                    Nonpolar neutral
        Valine Val    V aliphatic                    Nonpolar neutral
    Methionine Met    M aliphatic Nonpolar neutral, sulfur-containing
 Phenilalanine Phe    F  aromatic             Nonpolar neutral, large
   Thryptophan Trp    W  aromatic             Nonpolar neutral, large
      Tyrosine Tyr    Y  aromatic            Polar hydrophobic, large
    Asparagine Asn    N     polar                     Polar uncharged
     Glutamine Gln    Q     polar                     Polar uncharged
        Serine Ser    S     polar                     Polar uncharged
     Threonine Thr    T     polar                     Polar uncharged
      Arginine Arg    R     basic                         Charged (+)
     Histidine His    H     basic                         Charged (+)
        Lysine Lys    K     basic                         Charged (+)
     Aspartate Asp    D    acidic                         Charged (-)
     Glutamate Glu    E    acidic                         Charged (-)
       Glycine Gly    G      tiny          Hydrogen side chain (tiny)
       Proline Pro    P    cyclic             Cyclic nonpolar neutral
      Cysteine Cys    C   sbridge    Charged (-), SH-containing polar
Selenocysteine Sec    U  selenium   Charged (-), SeH-containing polar
   Pyrrolysine Pyr    O     amber   Charged (+), Bacteria and Archaea
```

#### 2.5. Contact spatial class

The last data structure is a table with two attributes, residue and class, reporting the spatial class of a given Ag residue. The current classes ("class1" to "class4") of an interaction specifies which part of the surface of the RBD is involved in the AAC formation. These classes must be defined empirically, based on RBD-FAB interaction 3D structures.

## 2. Ab-Ag complex prediction of affinity, stability, and health risk.

All the following steps can be easily executed through our [gbmContacts Jupyter notebook](https://github.com/fernandoPalluzzi/gbmContacts/blob/main/gbm_contacts.ipynb).

The first step is to load all the required datasets and generate an AbAgInteractionPredictor instance (input data tables can be passed as tab-separated text files or any data format convertible into a Pandas DataFrame):

```py
# Import required libraries.

import pandas as pd
from gbmContacts import RiskLevel, InteractionType, PredictionResult, AbAgInteractionPredictor

# Data loading.

contact_libs = "C:/Users/user/Desktop/gbmContacts/data/"

affinity = pd.read_csv(contact_libs + "Affinity_data.txt", sep = "\t")
metadata = pd.read_csv(contact_libs + "Affinity_meta.txt", sep = "\t")
contact_class = pd.read_csv(contact_libs + "Contact_class.txt", sep = "\t")
contact_map = pd.read_csv(contact_libs + "Contact_map.txt", sep = "\t")
mutations = pd.read_csv(contact_libs + "Contact_mutations.txt", sep = "\t")
properties = pd.read_csv(contact_libs + "Aminoacid_properties.txt", sep = "\t")

# AbAgInteractionPredictor instance.

abag = AbAgInteractionPredictor(
       affinity_data = affinity,
       affinity_metadata = metadata,
       contact_class = contact_class,
       contact_map = contact_map,
       contact_mutations = mutations,
       aa_properties = properties
)
```

Besides data loading, the AbAgInteractionPredictor builder restructure AAC data in convenient internal data formats. For example, the internal contact map lookup table is a default Python dictionary:

```py
print(abag.contact_map_lookup)
```
```
defaultdict(list,
            {('7kmg', 'h.R50'): ['V483', 'E484'],
             ('7kmg', 'h.L55'): ['L452', 'T470', 'F490'],
             ('7kmg', 'h.Y101'): ['E484', 'F490'],
             ('7kmg', 'h.R104'): ['Q493', 'S494'],
             ('7kmg', 'l.Y32'): ['F486', 'Y489'],
             ('7kmg', 'l.Y92'): ['F486', 'Y489'],
             ('7kmg', 'l.R96'): ['V483', 'E484'],
             ('7c01', 'h.S31'): ['Y473'],
             ('7c01', 'h.Y33'): ['K417', 'Y421', 'L455', 'F456'],
             ('7c01', 'h.Y52'): ['K417', 'Y421'],
             ('7c01', 'h.S53'): ['Y421', 'F456', 'Y473', 'Q493'],
             ...
             ('6zcz', 'h.W104'): ['S383'],
             ('6zcz', 'h.V105'): ['S383', 'K386'],
             ('6zcz', 'h.Y106'): ['S383', 'T385']})
```

The second step requires loading the query Ab-Ag complex that needs to be evaluated. It can be either specified manually as a Python Dict (see section 1) or from a plain text file where every line is a contact and each contact is a comma+space separated list of residues (the first is the Ab residue followed by the associated Ag residues). The gbmContacts library has a small builtin example (gbmContacts/data/unstable_complex_example.txt). After loading query data, we can simply launch model training and prediction:

```py
# Loading input contacts.
contacts = abag.load_contacts_from_file(contact_libs + "unstable_complex_example.txt")

# Interaction model training (done on the internal library).
abag_fit = abag.fit()

# Ab-Ag complex stability prediction (done on the query complex).
abag_predict = abag.predict(contacts)
```

The final output can be easily generated with:

```py
# Plot report in PNG format.
abag_trend = abag.plot_interaction_trend(abag_predict, output_file = "C:/Users/user/Desktop/AbAg_prediction_report.png")
```
<img width="5555" height="3148" alt="AbAg_prediction_report" src="https://github.com/user-attachments/assets/fde63429-4ef3-4d6f-b396-6f2f4995de58" />

As shown above, the results report is divided in 4 panels: **affinity profile**, **variant-associated risk**, **per-contact affinity**, **stability and confidence evaluation**. The affinity profile reports the predictions of every contact affinity (point estimate, average trend and 95% confidence interval), the overall Ab-Ag complex affinity (mean per-nanosecond predicted contact affinity), and the affinity threshold. The affinity threshold is the minimum affinity value that a contact must have to be considered stable, at any given time point. We wmpirically measured this value at 0.88, but can be changed when initializing the AbAgInteractionPredictor instance, using the affinity_threshold argument. The variant-associated risk depends on how stable the AAC is and increases exponentially with the time spent below the affinity threshold. The risk is a generic classification of how manageable is a new variant: up to 10% (manageable; there is at least one stable neutralizing Ab), from 10% to 30% (partially manageable; the contact is unstable and no neutralizing Ab is abailable, but based on stability there could be some attenuating interaction in a few available Ab), above 30% (unmanageable; any available Ab establish only extremely unstable contacts). The per-contact affinity barplot shows the stability connected to the best available antibody. If only one of these affinities falls below the threshold, the AAC is unstable, regardless the predicted profile. Finally, two more metrics are shown: stability (an overall stability evaluation score; >= 0.9: extremely stable, >= 0.8: stable but monitoring required; < 0.8: unstable) and confidence (how reliable is the stability prediction is; only meaningful if the AAC is stable; with unstable complexes the confidence tend to be naturally low).

Finally, if the reportlab library is installed, a more detailed PDF report can be generated, including plots and interpretations in English language:
```py
# Generate PDF report.
abag.generate_pdf_report(abag_predict, output_file = "C:/Users/user/Desktop/AbAg_prediction_report.pdf")
```
