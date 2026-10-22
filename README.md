# A51 simplicial arrangement

Exact verification and data for **“A sporadic simplicial arrangement of 51 lines”** by Tomás S. R. Silva, IMECC, Universidade Estadual de Campinas (UNICAMP).

```
@misc{silva2026sporadicsimplicialarrangement51,
      title={A sporadic simplicial arrangement of $51$ lines}, 
      author={Tomás S. R. Silva},
      year={2026},
      eprint={2610.28874},
      archivePrefix={arXiv},
      primaryClass={math.CO},
      url={https://arxiv.org/abs/2610.28874}, 
}
```

The construction is

$$
\mathcal A_{51}=I\cdot[1:0:0]\;\cup\;
I\cdot[0:\varphi:1]\;\cup\;I\cdot[0:1:s],
\qquad
\varphi=\frac{1+\sqrt5}{2},\quad s=\sqrt{\varphi}>0,
$$

where $I$ is the order-60 rotation group of the icosahedron. The three line orbits have sizes $15$, $6$, and $30$.

## Verify the paper

Python **3.9 or later**, standard library only:

```sh
python3 verify_a51.py
python3 verify_geometry.py
```

`verify_a51.py` is standalone and reads no data files. It constructs the rotation group, checks the closed coordinate formulas, computes all pairwise intersections and line–point incidences exactly, and verifies the twelve point orbits in Table 1. It checks simpliciality, irreducibility, the characteristic polynomial, and the numerical criteria used in the paper to establish sporadicity and nonsupersolvability.

The results are

$$
(t_2,t_3,t_5,t_6)=(120,190,36,15),\qquad
(f_0,f_1,f_2)=(361,1080,720),
$$

with all other intersection multiplicities zero, Melchior defect $0$, and characteristic polynomial $(T-1)(T^2-50T+669)$. Each original line contains $20$ vertices and each added line contains $22$.

`verify_geometry.py` imports the main verifier and checks the Section 4 formulas as identities in $K[t]$, the pencil involution, and the cross-ratio $s^3\notin\mathbb Q(\sqrt5)$. This obstruction concerns the given projective realization, without asserting rigidity.

Both scripts use exact rational arithmetic in $K=\mathbb Q[s]/(s^4-s^2-1)$, with no floating-point incidence tests. Failures return a nonzero exit status, including under `python3 -O`. Ordinary verification writes no files.

## SageMath notebook

Open [`a51_sagemath.ipynb`](a51_sagemath.ipynb) in Jupyter, select the **SageMath** kernel, and choose **Restart Kernel and Run All Cells**. Run it from the repository directory so it can compare the stored certificate. It requires SageMath with **PyNormaliz** and was fully executed with SageMath 10.8; saved outputs are included.

The notebook independently implements every computation in the paper, with explanations, the complete point-orbit table, and a hemisphere illustration. It uses Sage's native hyperplane-arrangement methods to verify the intersection lattice, characteristic polynomial, face vector, matroid connectivity, and all 1,440 central chambers (720 projective chambers). It also checks the geometric identities, both classical catalogue identifications, and every coordinate and incidence in the JSON certificate. It does not import the Python verifiers or write files.

## Exact data

[`data/certificate.json`](data/certificate.json) contains all 51 line normals, all 361 vertices and their incidences, the twelve point orbits, and the computed invariants. Reproduce it with:

```sh
python3 verify_a51.py --export-json data/certificate.json
```

Exports are deterministic. Each field coordinate is a list of four rational strings `["a0","a1","a2","a3"]`, representing $a_0+a_1s+a_2s^2+a_3s^3$. The interval $5/4<s<13/10$ selects the positive real root. Projective triples are normalized so their first nonzero coordinate is $1$.

The certificate uses schema version `1` and one-based line and vertex IDs. Lines are grouped as `O15`, `O6`, `O30`, then sorted by exact coordinates within each orbit; vertices are globally sorted. Point-orbit profiles count incident lines in that same orbit order.

An optional deletion check confirms that removing one representative line from each orbit gives nonsimplicial arrangements, with Melchior defects $12,10,12$:

```sh
python3 verify_a51.py --self-test
```

See [`LICENSE`](LICENSE) for the repository license.
