# Antimicrobial Resistance From Semantic Mutation Embedding (AIRFRAME)

## Overview

AIRFRAME leverages heterogeneous graph embedding to generate semantic representations of mutations, which are then used in either an embedding pooling method or a Transformer model to predict drug resistance and susceptibility. The pipeline includes:

1. Generating mutation embeddings using `metaPath2Vec.py`
2. Training a BERT model for drug resistance prediction using `BERT.py`

---

## Getting Started

Follow the steps below to set up the project and run the code.

### Step 1: Clone the Repository

To download the project from GitHub, run the following command:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name/codes
```

### Step 2: Set Up the Environment

#### Using `requirements.txt` with Conda

1. Create a new Conda environment:
   ```bash
   conda create --name airframe_env python=3.9
   conda activate airframe_env
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Code

### Generate Mutation Embeddings

To generate mutation embeddings, run `metaPath2Vec.py`:

```bash
python metaPath2Vec.py
```

This will output the mutation embeddings, typically saved in a file specified within the script.

### Train BERT for Drug Resistance Prediction

To train the BERT model, run `BERT.py`:

```bash
python BERT.py
```

This script will use the generated mutation embeddings and train a BERT model for drug resistance prediction. Ensure all necessary configuration parameters are set in the script or a configuration file.

---

## File Structure

```
.
├── codes/                # Directory for code scripts
│   ├── BERT.py           # Script for training BERT model
│   ├── metaPath2Vec.py   # Script for generating mutation embeddings
│   ├── modeling_bert.py  # Script for BERT model architecture
│   └── util.py           # Utility functions for the project
├── data/                 # Directory for input data and intermediate data
├── results/              # Directory for model outputs and results
├── requirements.txt      # Dependencies required for the project
├── README.md             # Project documentation
└── LICENSE.txt           # License for the project
```

---

## License

This project is licensed under the MIT License. See `LICENSE.txt` for more details.

---

## Contact

For any questions or feedback, contact emma002\@sjtu.edu.cn
