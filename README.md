# Paired-Contrastive-Feature-for-Highly-Reliable-Offline-Signature-Verification
This is part of the codes used in the paper titled **'Paired Contrastive Feature for Highly Reliable Offline Signature Verification.'**
**Paper link**: https://www.sciencedirect.com/science/article/pii/S0031320323005149?casa_token=QBCtBJqyGhgAAAAA:yK30z6yDbmDli5zLl3L_8BW9tnKKlKbKWGdxTbf8UF-QiXabCE9zk7Xc7fgFuS7CKiAJdalD

This project will keep being maintained and will receive regular updates.


## Requirements
- TensorFlow
- Keras
- python3

## Datasets
- **'BHSig260'**: can be downloaded from http://www.gpds.ulpgc.es/download
  
- **'UTSig'**: can be downloaded from http://mlcm.ut.ac.ir/Datasets.html
  
- Feel free to experiment with other offline signature datasets if you have them.

## How to run SigNet_train_and_evaluate?
Execute the following command:
```
python ./SigNet/SigNet_script.py
```

## Update till now
- './Data/process_Bengali.py': Example codes for processing the signature dataset and generating paired signature names.

- './SignatureDataGenerator.py': Codes for generating data for training, validation, and testing.

- './datasets_information.py': Codes containing basic information about datasets.
  
- './SigNet/': Codes related to SigNet.
  - The original codes are sourced from the paper 'SigNet: Convolutional Siamese Network for Writer Independent Offline Signature Verification' with code 'https://github.com/sounakdey/SigNet'
 
- './Top_rank/': Codes related to Top-rankNN.

## Comments
The 'Selective_rejection' file is currently under refinement.

For reference, you can check the original paper 'SelectiveNet: A Deep Neural Network with an Integrated Reject Option' and its associated codes: https://github.com/geifmany/selectivenet
