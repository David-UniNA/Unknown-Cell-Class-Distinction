Open-set recognition of cell snapshots
Neural network-based image classification is widely used in life science applications. However, it is essential to extrapolate a correct classification method for unknown images, where no prior knowledge can be utilised. Under a closed-set assumption,

![image](https://github.com/user-attachments/assets/3e21b71f-e920-4b0e-b7a2-163a3789169e)
unknown images (black dots) will be inevitably misclassified, but this can be genuinely overcome choosing an open-set classification approach,

![image](https://github.com/user-attachments/assets/50b4072c-5fe0-45cb-b823-1062be6d7b41)
which first generates an in-distribution of identified images to successively discriminate out-of-distribution images. The testing of such image classification for single cell applications in life science scenarios has yet to be done but could broaden our expertise in quantifying the influence of prediction uncertainty in deep learning. In this framework, we implemented the open-set concept on scattering snapshots of living cells to distinguish between unknown and known cell classes, targeting four different known monoblast cell classes (un-polarized, pro-inflammatory and anti-inflammatory Macrophages as well as Monocytes) and a single tumoral unknown monoblast cell line (THP1).


Citations:
    If you use any methods, datasets, or specific algorithms, please cite.
    
    Cioffi G, Dannhauser D, et al. Biomed Opt Express., 14(10):5060-5074, 2023. doi: 10.1364/BOE.492028
