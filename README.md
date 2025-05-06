
TODO: revise readme
# Counterfactual Routing Competition (CRC)

Welcome to the official repository for the **Counterfactual Routing Competition (CRC)**, held at [IJCAI 2025](https://2025.ijcai.org/), the premier international conference for artificial intelligence research. CRC challenges participants to generate insightful, counterfactual explanations for personalized route planning.

---

## Table of Contents

- [About the Competition](#about-the-competition)
- [Competition Format](#competition-format)
- [Important Dates \& Submission Guidelines](#important-dates--submission-guidelines)
- [Getting Started](#getting-started)
- [Baseline Solution](#baseline-solution)
- [Requirements](#requirements)
- [Contact](#contact)

---

## About the Competition

The CRC is part of IJCAI 2025, taking place in Montreal, Canada, from August 16–22, 2025. For more details and updates, please visit the competition website: **[TBD]**.

This competition focuses on explainable AI in the context of personalized route planning. Participants will be tasked with answering the following user-centric question:

> **"Why is the computed (fact) route optimal for me, and not the alternative (foil) route?"**

Your goal is to generate a *counterfactual map*: a minimally modified version of the original map where the foil route becomes optimal for the given user, according to their personalized preferences.

---

## Competition Format

Participants will be provided with:

- **A map**: The environment for route planning.
- **A route planner**: Computes optimal routes based on map and user preferences.
- **Start and destination locations**: The endpoints for routing.
- **A user model**: Encodes individual routing preferences (assumed correct for the competition).

**Key Concepts:**

- **Fact Route**: The optimal route for the user on the original map.
- **Foil Route**: An alternative route between the same start and destination.
- **Counterfactual Map**: A map as similar as possible to the original, where the foil route becomes optimal for the user.
- **Similarity**: Measured using a graph distance metric (details provided in the competition materials).
- **Map Operators**: The allowed modifications to the map for generating counterfactuals.

*Note*: Solutions where the foil route is nearly optimal (but not identical) may still be considered valid, as long as the changes are minimal and justified.

---

## Important Dates \& Submission Guidelines

**Key Dates:**

- **Competition Submission Deadline:** July 10, 2025
- **Competition Results Release:** July 30, 2025
- **Announcement of Presentation Slots:** August 5, 2025
- **Results Presentation and Winner Announcement:** at IJCAI 2025

*All submission deadlines are at 23:59h AoE (Anywhere on Earth).*

**Submission Instructions:**

- Please send your contribution via: **[TBD]**
- Each team must submit:
    - **Code base** for your solution
    - **Report (2–4 pages)** in the default IJCAI 2025 template, following academic standards.
---

## Getting Started

### Requirements

- **Python**: Version 3.8 or higher

To install the required packages, run:

```bash
pip install -r requirements.txt
```

---

## Baseline Solution

A baseline implementation using local search is provided in the repository:

- **Notebook**: `ls_demon.ipynb`

This notebook demonstrates a simple approach for generating counterfactual maps. Participants are encouraged to use this as a starting point and improve upon it.

---

## Contact

For questions or further information, please refer to the competition website (**[TBD]**) or contact the organizers.



---

*We look forward to your participation and innovative solutions at CRC, IJCAI 2025!*

---
This competition is organised in the context of the European PEER project, and includes organisers from the municipality of Amsterdam. The maps/graphs are based on real-world data and were created by the municipality of Amsterdam, using various techniques including point-cloud based obstacle and width measurements and recognising suitable crossings from satellite imagery (Link1, Link2). The competition focuses on key AI components of explainable routing by providing counterfactual explanations, and its results may be used (attributed of course) in a demonstrator for accessible route planning. The competition is run open-source.


