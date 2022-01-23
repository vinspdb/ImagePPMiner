# Using Convolutional Neural Networks for Predictive Process Analytics

**The repository contains code referred to the work:**

*Vincenzo Pasquadibisceglie, Annalisa Appice, Giovanna Castellano, Donato Malerba*


[*Using Convolutional Neural Networks for Predictive Process Analytics*](https://ieeexplore.ieee.org/document/8786066)

Please cite our work if you find it useful for your research and work.

```
@INPROCEEDINGS{8786066,
  author={V. {Pasquadibisceglie} and A. {Appice} and G. {Castellano} and D. {Malerba}},
  booktitle={2019 International Conference on Process Mining (ICPM)}, 
  title={Using Convolutional Neural Networks for Predictive Process Analytics}, 
  year={2019},
  volume={},
  number={},
  pages={129-136},
  doi={10.1109/ICPM.2019.00028}}
```
# How to use

Train neural network
- event_log: event log name
- n_layers: number of convolutional layers

```
python main.py -event_log helpdesk -n_layers 2
```
