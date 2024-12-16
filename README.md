# Clustering by Mining Density Distributions and Splitting Manifold Structure
This is the original implementation of the algorithm MDMSC in the paper "Clustering by Mining Density Distributions and Splitting Manifold Structure" accepted in AAAI-25. 

# Files
The program mainly containing:
* synthetic datasets (synthesis) are in txt format.
* real world datasaets consist of four txt files (arff_txt) and nine mat files (real_all).
* three python files (evaluation.py、MDMSC.py and main.py)
* requirements
* README.md

# Requirements
pip install -r requirements.txt
* matplotlib==3.5.1
* networkx==3.2.1
* numpy==1.22.3
* scikit-learn==1.0.2
* scipy==1.8.0
```
pip install -r requirements.txt
```
# Usage
```
python main.py
```
 Run the main.py script and obtain the results.
 Before running, you can choose the syn function or real function to cluster the synthetic data or real world data.

# Citation
If you find this file useful in your research, please consider citing:
@inproceedings{xu2025clustering,
      title={Clustering by Mining Density Distributions and Splitting Manifold Structure}, 
      author={Zhichang Xu and Zhiguo Long and Hua Meng},
      year={2024},
      pages={accpeted in AAAI-25} 
}