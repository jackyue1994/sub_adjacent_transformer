# Sub-Adjacent Transformer (IJCAI 2024)
Sub-Adjacent Transformer: Improving Time Series Anomaly Detection with Reconstruction Error from Sub-Adjacent Neighborhoods

We present the Sub-Adjacent Transformer with a novel attention mechanism for unsupervised time series anomaly detection. Unlike previous approaches that rely on all the points within some neighborhood for time point reconstruction, our method restricts the attention to regions not immediately adjacent to the target points, termed sub-adjacent neighborhoods. Our key observation
is that owing to the rarity of anomalies, they typically exhibit more pronounced differences from their sub-adjacent neighborhoods than from their immediate vicinities. By focusing the attention on
the sub-adjacent areas, we make the reconstruction of anomalies more challenging, thereby enhancing their detectability.

<p align="center">
<img src=".\pics\structure_ours.png"  alt="" align=center />
</p>

## Get Started

1. Install necessary packages listed in requirement.txt.
2. Download data. You can obtain four benchmarks from [Google Cloud](https://drive.google.com/drive/folders/1gisthCoE-RrKJ0j3KPV7xiibhHWT9qRm?usp=sharing). **All the datasets are well pre-processed**. For the SWaT dataset, you can apply for it by following its official tutorial.
3. Train and evaluate. We provide the experiment scripts of all benchmarks under the folder `./scripts`. You can reproduce the experiment results as follows:
```bash
bash ./scripts/SMD.sh
bash ./scripts/MSL.sh
bash ./scripts/SMAP.sh
bash ./scripts/PSM.sh
```

Especially, we use the adjustment operation proposed by [Xu et al, 2018](https://arxiv.org/pdf/1802.03903.pdf) for model evaluation. 

## Main Result

We compare our model with 15 baselines, including TimesNet, Anomaly Transformer, etc. **Generally, our method achieves SOTA.**

<p align="center">
<img src=".\pics\result_sa.png"  alt="" align=center />
</p>

## Citation
If you find this repo useful, please cite our paper. 

```
@article{yue2024sub,
  title={Sub-Adjacent Transformer: Improving Time Series Anomaly Detection with Reconstruction Error from Sub-Adjacent Neighborhoods},
  author={Wenzhen Yue and Xianghua Ying and Ruohao Guo and DongDong Chen and Ji Shi and Bowei Xing and Yuqing Zhu and Taiyan Chen},
  journal={arXiv preprint arXiv:2404.18948},
  year={2024}
}
```

## Contact
If you have any question, please contact yuewenzhen@stu.pku.edu.cn.
