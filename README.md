# `ml-framework-scaffold`

## Overview

`ml-framework-scaffold` is a lightweight, production-ready Machine Learning project structure built for clarity, modularity, and scalability. It includes a `Hello World` RandomForestClassifier model using the Iris dataset, and is ideal for data science practitioners, ML engineers, or students who want a clean slate to start any ML project.

---

## Prerequisites

- **Python**: 3.10+
- **Conda** (recommended) or `venv` for virtual environments

To verify you have the required tools installed:

```bash
python --version
conda --version   # or skip if using venv
```

---

## Installation

Choose one of the following methods:

### Option 1: Using Conda (Recommended)

```bash
# Create a new environment
conda create -n ml_hello_env python=3.10 -y

# Activate the environment
conda activate ml_hello_env

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using venv

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 3: Global Install (Not Recommended)

```bash
pip install -r requirements.txt
```

> **Note**: Global installs can cause dependency conflicts with other projects.

---

## Usage

Run the project with:

```bash
python main.py
```

This will:

- Load the Iris dataset  
- Train a RandomForestClassifier  
- Print the accuracy score  

---

## Project Structure

```bash
ml-hello-world/
├── data/                  # Raw, processed, and external data
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/             # Jupyter notebooks
├── src/                   # Source code
│   ├── data/              # Data loading and preprocessing
│   ├── features/          # Feature engineering
│   ├── models/            # Training and evaluation
│   ├── utils/             # Helper utilities
│   └── visualization/     # Plotting and graphs
├── tests/                 # Unit tests
├── main.py                # Main entry point
├── requirements.txt       # Dependency list
└── README.md              # You're here!
```

---

## Platform-Specific Instructions

### macOS / Linux / WSL

- Clone the repository.
- Use Conda or venv to set up the environment.
- Run `main.py`.

### Windows

Follow the same steps. To activate the environment:

```bash
.venv\Scripts\activate
```

---

## Future Enhancements

- Add pytest test coverage  
- Config-driven training with YAML  
- Docker support for reproducibility  
```

