# LLM4AMC: Adapting Large Language Models for Adaptive Modulation and Coding  

This repository will contain the **implementation** of our paper:  

> **LLM4AMC: Adapting Large Language Models for Adaptive Modulation and Coding**  
> *X. Pan, B.Liu, X.Cheng and C.Chen*  
> Accepted by Journal of Communications and Information Networks (JCIN), 2025.

---

## Overview

Adaptive modulation and coding (AMC) is a key technology in 5G new radio (NR), enabling dynamic link adaptation by balancing transmission efficiency and reliability based on channel conditions. However, traditional methods often suffer from performance degradation due to the aging issues of channel quality indicator (CQI). Recently, the emerging capabilities of large language models (LLMs) in contextual understanding and temporal modeling naturally align with the dynamic channel adaptation requirements of AMC technology. Leveraging pretrained LLMs, we propose a channel quality prediction method empowered by LLMs to optimize AMC, termed LLM4AMC. We freeze most parameters of the LLM and fine-tune it to fully utilize the knowledge acquired during pretraining while better adapting it to the AMC task. We design a network architecture composed of four modules, a preprocessing layer, an embedding layer, a backbone network, and an output layer, effectively capturing the time-varying characteristics of channel quality to achieve accurate predictions of future channel conditions.

---


## Contact
  
For any questions, please contact: [2501213461@stu.pku.edu.cn]  

---

## Citation

If you use this work, please cite:

```bibtex
@article{LLM4AMC_2025,
  title={LLM4AMC: Adapting Large Language Models for Adaptive Modulation and Coding},
  author={X.Pan, B.Liu, X.Cheng and C.Chen},
  journal={Journal of Communications and Information Networks},
  year={2025}
}

