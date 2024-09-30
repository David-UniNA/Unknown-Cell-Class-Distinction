Open-set recognition of cell snapshots via auxillary open set risk (AOSR)


Neural Networks show high prediction performance for known classes (aslo called closed-set assumption), but struggles when unknown image classes need to be identified. In such a scenario no prior knowledge of the unknown image class can be used for the model training, which inevitably results in a misclassification. 

![image](https://github.com/user-attachments/assets/3e21b71f-e920-4b0e-b7a2-163a3789169e) (Misclassification of unknown image class, black dots)

To overcome the hurdle, of identifying unknown cell classes, we must first define an in-distribution of known classes to afterwards detect out-of-distribution images as unknowns. 

![image](https://github.com/user-attachments/assets/50b4072c-5fe0-45cb-b823-1062be6d7b41) (Out-of-distibution area for unknown image destinction)

Ones, such a new image class is identified, we can retrain our image classifier with the obtained knowledge, so we dynamically update the image class database. We applied this measurement approach to scattering pattern snapshots (images) of different classes of living cells to distinguish between unknown and known cell classes, targeting four different known monoblast cell classes (un-polarized, pro-inflammatory and anti-inflammatory Macrophages as well as Monocytes) and a single tumoral unknown monoblast cell line (THP1).



If you use any methods, datasets, or specific algorithms, please cite the following work.

    Cioffi G, Dannhauser D, et al. Biomed Opt Express., 14(10):5060-5074, 2023. doi: 10.1364/BOE.492028
