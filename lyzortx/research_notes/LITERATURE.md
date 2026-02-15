### **Curated Reading List: Phage-Host Prediction**

This list is organized into three categories: foundational machine learning approaches, advanced deep learning and network methods, and key review articles that provide a broad overview of the field.

---

#### **1. Foundational Machine Learning & Feature-Based Methods**

These papers represent the core techniques of using genomic features with classical ML models like Random Forests and SVMs.

*   **Paper:** **WIsH: who is the host? Predicting prokaryotic hosts from metagenomic phage contigs**
    *   **Authors:** Galiez, C., et al. (2017)
    *   **Journal:** *Bioinformatics*
    *   **Key Method:** Uses k-mer frequencies (specifically, oligonucleotide usage) to calculate a distance score between phage and potential hosts. It's a foundational "genomic similarity" approach.
    *   **Relevance:** Directly relates to using k-mer features, a likely strong signal for our model.

*   **Paper:** **PHISDetector: A tool to detect prophage-host sequences using deep learning**
    *   **Authors:** Tovar-Herrera, O.E., et al. (2021)
    *   **Journal:** *PeerJ*
    *   **Key Method:** Combines multiple feature types (sequence similarity, CRISPR matches, tRNA presence) into a Random Forest model to achieve high accuracy.
    *   **Relevance:** Provides a blueprint for integrating diverse feature sets (like our Tracks C, D, and E) into a single, powerful model.

*   **Paper:** **VirHostMatcher: Predicting phage-host relationship using blasting and host-related mobile genetic elements**
    *   **Authors:** Ahlgren, N.A., et al. (2016)
    *   **Journal:** *Bioinformatics*
    *   **Key Method:** Leverages sequence homology (BLAST) and shared mobile genetic elements (like prophages and CRISPR spacers) to link phages to hosts.
    *   **Relevance:** Validates the importance of CRISPR spacers and other genomic "footprints" as predictive features.

#### **2. Advanced Deep Learning & Network Approaches**

These papers explore more recent, complex models that learn feature representations automatically.

*   **Paper:** **Predicting phage-host interactions with deep learning and natural language processing**
    *   **Authors:** Vanni, C., et al. (2022)
    *   **Journal:** *iScience*
    *   **Key Method:** Treats protein sequences as "sentences" and applies Natural Language Processing (NLP) models (like Doc2Vec) to learn embeddings for both phage and host proteins, predicting interactions based on their similarity in this learned space.
    *   **Relevance:** Highly relevant for our Feature Engineering tracks (C & D), suggesting a powerful way to represent protein features like RBPs and receptors without manual domain annotation.

*   **Paper:** **VIRHostMatcher-Net: a graph-based deep learning method for predicting phage-host interactions**
    *   **Authors:** Shang, J., et al. (2021)
    *   **Journal:** *Bioinformatics*
    *   **Key Method:** Represents the entire phage-host ecosystem as a graph and uses Graph Convolutional Networks (GCNs) to learn patterns and predict missing links (interactions).
    *   **Relevance:** While potentially beyond the initial scope, this represents the state-of-the-art and could inform future directions for modeling complex, multi-phage, multi-host relationships.

#### **3. Key Reviews and Perspectives**

These articles provide excellent summaries of the field, common challenges, and benchmark comparisons.

*   **Paper:** **A critical assessment of computational tools for phage-host prediction**
    *   **Authors:** Edwards, R.A., et al. (2016)
    *   **Journal:** *FEMS Microbiology Reviews*
    *   **Key Method:** A comprehensive review comparing the performance of various early-generation prediction tools on a standardized dataset.
    *   **Relevance:** Essential for understanding the history of the field and for learning from the strengths and weaknesses of previous methods. It helps justify our plan to build a custom, integrated pipeline.
