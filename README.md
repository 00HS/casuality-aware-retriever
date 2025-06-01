# casuality-aware-retrieval

Official implementation of the paper:  
**[Causal Retrieval with Semantic Consideration](https://arxiv.org/abs/2504.04700)** 

<p align="center">
      <img src="causal_retrieval.png" alt="Causal Retrieval Overview" width="600"/>
</p>

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/00HS/casuality-aware-retrieval.git
cd casuality-aware-retrieval
```

### 2. Train the model

```bash 
bash run_train.sh
```
If you'd like to use a checkpoint, please refer to our Hugging Face repository:  
[0Hyun/Cawai on Hugging Face](https://huggingface.co/0Hyun/Cawai)

### 3. Evaluate the model

```bash 
bash run_eval.sh
```

## Roadmap

We are planning to support the following datasets and features:
- [ ] **ScienceQA**
- [ ] **GeneralQA**

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{shin2025causalretrievalsemanticconsideration,
      title={Causal Retrieval with Semantic Consideration}, 
      author={Hyunseo Shin and Wonseok Hwang},
      year={2025},
      eprint={2504.04700},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.04700}, 
}
```