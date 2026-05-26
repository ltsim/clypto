# mealpy-legacy-collection

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)
![PyPI - Version](https://img.shields.io/pypi/v/mealpy-legacy-collection?style=flat-square)
![PyPI - Implementation](https://img.shields.io/pypi/implementation/mealpy-legacy-collection?style=flat-square)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mealpy-legacy-collection?style=flat-square)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/mealpy-legacy-collection?style=flat-square)
![GitHub Release Date](https://img.shields.io/github/release-date/ltsim/mealpy-legacy-collection.svg?style=flat-square)
![PyPI - Downloads](https://img.shields.io/pypi/dm/mealpy-legacy-collection?style=flat-square)

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/ltsim/mealpy-legacy-collection/publish.yml?style=flat-square&logo=pypi&label=Publish)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/ltsim/mealpy-legacy-collection/test.yml?style=flat-square&logo=pytest&label=Testing)

Is a lightweight, heavily refactored fork of [MEALPY (MEta-Heuristic Algorithms using PYthon)](https://github.com/thieu1995/mealpy), stripped down to its absolute mathematical core. It serves as a pure, bloat-free repository that preserves the historical collection of population-based metaheuristic algorithms (PBM), acting as an open architectural catalog for educational purposes, clean research, and rapid prototyping within our ecosystem.

Unlike monolithic optimization frameworks, **mealpy-legacy-collection** cuts out all secondary overhead such as visualization tools, complex logging, and heavy external dependencies focusing strictly on the raw algorithmic logic and mathematical transition operators of these legacy implementations. This ensures a clean, decoupled foundation for developers to study, test, and benchmark classic metaheuristics without the friction of modern software bloat.

## Why?

* **Lean Core & Zero Bloat:** Amputated all UI, plotting, and file-writing features to create an ultra-lightweight library, making this massive historical catalog ideal to be consumed as a clean, plug-and-play mathematical dependency.
* **Aggressive Agent Refactoring:** Redesigned the original agent state model to eliminate heavy Python object/dictionary overhead, shifting towards a streamlined, flat population structure that maximizes data locality.
* **Didactic Catalog Architecture:** Re-engineered with a highly pedagogical layout. Students and researchers can open any legacy algorithm, understand its transition rules in pure Python/NumPy in just a few lines of code, and easily use the structure as a mold to study or adapt classic metaheuristics.
* **Ecosystem Ready:** Built to operate as a decoupled, standardized engine, making this collection perfectly tailored to be driven by external benchmarking, evaluation frameworks, and modern optimization wrappers.

## Credits & Citation Request

This project is based is entirely built upon the foundational work, dedication, and effort of the original [MEALPY](https://github.com/thieu1995/mealpy) authors, **Nguyen Van Thieu** and **Seyedali Mirjalili**, as well as the global community of researchers and scientists who originally designed, investigated, and implemented the diverse metaheuristic algorithms contained within this collection. This fork merely restructures their brilliant mathematical work for decoupled, lightweight, and didactic purposes.

If you use this library, its restructured architecture, or the underlying algorithms in your academic research, please ensure proper credit is given to the original creators by citing:
```bibtex 
@article{van2023mealpy,
  title={MEALPY: An open-source library for latest meta-heuristic algorithms in Python},
  author={Van Thieu, Nguyen and Mirjalili, Seyedali},
  journal={Journal of Systems Architecture},
  year={2023},
  publisher={Elsevier},
  doi={10.1016/j.sysarc.2023.102871}
}

@article{van2023groundwater,
  title={Groundwater level modeling using Augmented Artificial Ecosystem Optimization},
  author={Van Thieu, Nguyen and Barma, Surajit Deb and Van Lam, To and Kisi, Ozgur and Mahesha, Amai},
  journal={Journal of Hydrology},
  volume={617},
  pages={129034},
  year={2023},
  publisher={Elsevier},
  doi={https://doi.org/10.1016/j.jhydrol.2022.129034}
}

@article{ahmed2021comprehensive,
  title={A comprehensive comparison of recent developed meta-heuristic algorithms for streamflow time series forecasting problem},
  author={Ahmed, Ali Najah and Van Lam, To and Hung, Nguyen Duy and Van Thieu, Nguyen and Kisi, Ozgur and El-Shafie, Ahmed},
  journal={Applied Soft Computing},
  volume={105},
  pages={107282},
  year={2021},
  publisher={Elsevier},
  doi={10.1016/j.asoc.2021.107282}
}
```

## Algorithmic Citations
To honor the individual authors who contributed each method to the field of approximate optimization, we strongly encourage you to cite the specific foundational papers for the algorithms used in your experiments.

You can find the complete list of original scientific papers and their corresponding citations in our dedicated [REFERENCES.md](REFERENCES.md) file included within this repository.

## Goals
The sole purpose of this repository is to preserve a clean, decoupled collection of both classical and state-of-the-art nature-inspired metaheuristic algorithms. By stripping away all monolithic overhead, our goal is to provide a standardized, raw algorithmic library that can be effortlessly integrated, embedded, and reused across any Python environment—whether for academic research, proprietary commercial software, or high-performance computing clusters. 

### What you can do

- **Universal Reusability:** Import and execute any metaheuristic algorithm across any third-party framework or custom pipeline without dependency conflicts.
- **Pure Logic Inspection:** Open and study the raw mathematical transition operators of a massive catalog of population-based algorithms.
- **Zero-Friction Integration:** Consume the entire collection as a lightweight, plug-and-play mathematical engine.
- **Benchmark-Ready Sourcing:** Use these legacy implementations as standardized baselines for modern benchmarking and comparative evaluation suites.

# Usage

## Installation

* Install the stable (latest) version from [PyPI release](https://pypi.python.org/pypi/mealpy-legacy-collection):
```bash
$ pip install mealpy-legacy-collection --upgrade
```

* Install the pre-release version directly from the source code:
```bash
$ git clone https://github.com/ltsim/mealpy-legacy-collection.git
$ cd mealpy-legacy-collection
$ python -m pip install .
```

* In case, you want to install the development version from Github:
```bash
$ pip install git+https://github.com/ltsim/mealpy-legacy-collection.git 
```

## Optimizer Classification Table

* Meta-heuristic Categories: ([Based on this article](https://doi.org/10.1016/j.procs.2020.09.075))
    + Evolutionary-based: Algorithms inspired by Darwin's law of natural selection and evolutionary computing principles
    + Swarm-based: Algorithms drawing inspiration from the collective movement and interaction of swarms (e.g., birds, social insects).
    + Physics-based: Algorithms derived from physical laws and phenomena (e.g., Newton's law of universal gravitation, black holes, multiverse theory).
    + Human-based: Algorithms inspired by human interactions and behaviors (e.g., queuing search, teaching-learning processes).
    + Biology-based: Algorithms based on biological creatures or microorganisms.
    + System-based:  Algorithms inspired by ecological systems, immune systems, or network systems.
    + Math-based: Algorithms developed from mathematical forms or laws (e.g., sine-cosine functions).
    + Music-based: Algorithms drawing inspiration from musical instruments or compositions.

* Difficulty - Difficulty Level (Personal Opinion): **Objective observation from author**. Depend on the number of 
  parameters, number of equations, the original ideas, time spend for coding, source lines of code (SLOC).
    + Easy: A few paras, few equations, SLOC very short
    + Medium: more equations than Easy level, SLOC longer than Easy level
    + Hard: Lots of equations, SLOC longer than Medium level, the paper hard to read.
    + Hard* - Very hard: Lots of equations, SLOC too long, the paper is very hard to read.

For newbie, we recommend to read the paper of algorithms which difficulty is "easy" or "medium" difficulty level.

### Warning: Algorithms Suspected of Plagiarism

During our implementation and classification of metaheuristic optimization algorithms, we identified a set of methods that raise 
serious concerns regarding **scientific integrity and originality**. These algorithms are typically published under **different names**, 
but they appear to share:

- The **same core mathematical models**, equations, and update rules.
- Only superficial changes in naming, metaphors, or biological analogies.
- Publications authored by **the same or overlapping research groups**.
- **Heavy criticism** on public academic forums such as [PubPeer](https://pubpeer.com), where many of these papers are flagged for **self-plagiarism**, **redundant publication**, or **lack of novelty**.
- Some of these papers may be **withdrawn or retracted in the future**, as investigations unfold.

For these reasons, we strongly advise the **exclusion** of the following algorithms from scientific benchmarking, 
comparative studies, or any applications unless their originality is transparently validated.

**I have personally implemented these algorithms, which is why I can confidently say that they are nearly identical 
and likely cases of plagiarism. For this reason, I will no longer spend time coding such algorithms in the future. 
This warning is intended to help others avoid using or relying on these methods in their work.**

### Ethical Reminder

Researchers and students are urged to **exercise caution** when referencing or applying the algorithms listed above. 
Using unoriginal or unethical work can compromise the **scientific credibility** of any downstream research and introduce **misleading experimental results**.

> **Check [PubPeer1](https://pubpeer.com/publications/1F5DCE5BC42BF2D77A1B0C281A5295)** and [PubPeer2](https://pubpeer.com/publications/D47357D409AE273F9E03C7CBE30EB7) to 
> find ongoing discussions and critiques from the academic community.

For detailed information about the updates in each new version, see the [ChangeLog](/CHANGELOG.md) file.

---

* Maintained by: [LTSIM](mailto:tsim@cucei.udg.mx) @ 2026
* Developed by: [Thieu](mailto:nguyenthieu2102@gmail.com?Subject=Opfunu_QUESTIONS) @ 2023
