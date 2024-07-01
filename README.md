<br>
<p align="center">
<h1 align="center"><strong>Empowering 3D Visual Grounding with Reasoning Capabilities</strong></h1>
  <p align="center">
  	<strong>ECCV 2024</strong>
	<br>
    <a href='https://zcmax.github.io//' target='_blank'>Chenming Zhu</a>&emsp;
	<a href='https://tai-wang.github.io/' target='_blank'>Tai Wang</a>&emsp;
    <a href='https://zhangwenwei.cn/' target='_blank'>Wenwei Zhang</a>&emsp;
    <a href='https://chenkai.site/' target='_blank'>Kai Chen</a>&emsp;
	<a href='https://xh-liu.github.io//' target='_blank'>Xihui Liu</a>&emsp;
    <br>
    Shanghai AI Laboratory&emsp;The University of Hong Kong
    <br>
  </p>
</p>


<div id="top" align="center">

[![](https://img.shields.io/badge/Paper-%F0%9F%93%96-blue)](./assets/ECCV_2024_ScanReason.pdf)
[![](https://img.shields.io/badge/Project-%F0%9F%9A%80-blue)](https://zcmax.github.io/projects/ScanReason/)

</div>


## 🏠 Background
<!-- ![Teaser](assets/teaser.jpg) -->

<div style="text-align: center;">
    <img src="assets/Fig_Teaser.png" alt="Dialogue_Teaser" width=100% >
</div>
For an embodied agent, they not only need to be able to understand the 3D
environment and complex human instructions, but also localize the target objects for
interaction and navigation. Although GPT-4 (GPT-4V) have strong text (multi-modal)
reasoning abilities, they lack the ability to directly perceive the 3D scene, understand
the 3D spatial relationships and output corresponding target object locations. In this paper, we propose a new task called 3D reasoning grounding and introduce a new benchmark **ScanReason** which provides over 10K question-answer-location pairs from five reasoning types that require the synerization of reasoning and grounding. We further design our approach, ReGround3D, composed of the visual-centric reasoning module empowered by Multi-modal Large Language Model (MLLM) and the 3D grounding module to obtain accurate object locations by looking back to the enhanced geometry and fine-grained details from the 3D scenes. A chain-of-grounding mechanism is proposed to further boost the performance with interleaved reasoning and grounding steps during inference. 

## 📦 Benchmark and Model
### Benchmark Overview
<p align="center">
  <img src="assets/scanreason_benchmark.png" align="center" width="100%">
</p>
ScanReason is the first comprehensive and hierarchical 3D reasoning grounding benchmark. We define 5 types of questions depending on which type of reasoning is required: Spatial reasoning and function reasoning require fundamental understanding of the 3D physical world, focusing on objects themselves and inter-object spatial relationships in a 3D scene respectively, and logistic reasoning, emotional reasoning, and safety reasoning are high-level reasoning skills built upon the two fundamental reasoning abilities to address user-centric real-world applications.

<p align="center">
  <img src="assets/3d_reasoning_grounding.png" align="center" width="100%">
</p>

### Model Overview
<p align="center">
  <img src="assets/Fig_Method.png" align="center" width="100%">
</p>

## 📝 TODO List

- \[x\] First Release.
- \[ \] Release ScanReason datasets and benchmark.
- \[ \] Release ReGround3D code.


## 📄 License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>
<br />
This work is under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

## 👏 Acknowledgements

This repo benefits from [LISA](https://github.com/dvlab-research/LISA), [EmbodiedScan](https://github.com/OpenRobotLab/EmbodiedScan), [3D-LLM](https://github.com/UMass-Foundation-Model/3D-LLM), [LLaVA](https://github.com/haotian-liu/LLaVA). 