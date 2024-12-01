Enhancing Underwater Images via Asymmetric Multi-Scale Invertible Networks
=
This is the official PyTorch implementation of the ACMMM 2024 paper.

Data
-
Put the training data you need under the directory 'data'.

Rename the directories that contains underwater images and enhanced images to 'raw' and 'reference'.

Train
-
Set the training configs in train_config.py;

Set the root of the training data in train.py by function Dataset();

If you want to continue training on a trained model, remember to reset the resume, resume_epoch and resume_optimizer in train_config.py.

Test
-
Set your traied model in test.py by function load_model();

Set the root of your testing data in test.py.

Citation
-
    @inproceedings{quan2024enhancing,
      title={Enhancing Underwater Images via Asymmetric Multi-Scale Invertible Networks},
      author={Quan, Yuhui and Tan, Xiaoheng and Huang, Yan and Xu, Yong and Ji, Hui},
      booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
      pages={6182--6191},
      year={2024}
    }

Contacts
-
If you have questions, please contact with csxiaohengtan@foxmail.com.
