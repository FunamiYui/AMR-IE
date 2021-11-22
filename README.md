# AMR-IE
The code repository for [AMR guided joint information extraction model](https://www.aclweb.org/anthology/2021.naacl-main.4/) (NAACL-HLT 2021). 

## AMR Parser
We use the transformer-based AMR parser [(Astudillo et al. 2020)](https://www.aclweb.org/anthology/2020.findings-emnlp.89/) in our model. We train the AMR parser on LDC AMR 3.0 data, and the [code](https://drive.google.com/file/d/1SB36NyEaRd740rGTjD_8ga7l5NGeRlkR/) and the [pretrained model](https://drive.google.com/file/d/1LRJuOwHQ6EWmzRBpYpWwsr_5m2kO-IP7) are publicly available. Please follow the README file in the code directory to install and use it.

The code of the AMR parser is derived from the original [Github Repo](https://github.com/IBM/transition-amr-parser). We made some slight modifications to fit it in higher versions of PyTorch. 

## Environments
The core python dependencies for this code are as follows:

`torch==1.6.0`, `torch-scatter==2.0.6`, `transformers==4.3.0`, `dgl==0.6.0`.

For other dependencies, you can simply use `pip install` to install the required packages that are indicated by the error messages.

And for IBM/transition-amr-parser's dependencies, refer to this [link](https://github.com/IBM/transition-amr-parser/tree/stack-transformer), DO NOT `pip install` directly.

关于使用IBM的pretrained-parser（stack-transformer）有一些需要记录的点：
- 目前已经可以使用，并跑通了对oneie数据的parse。
- 原repo是没有也不会放出来pretrained-model的，因为issue里面他们说训练用的AMR LDC数据集是非公开的，所以不放。
- 但是AMR-IE repo放出来了pretrained-parser，包括所使用的code和pretrained-model(checkpoint)，pretrained-model是在AMR-IE作者在AMR3.0上训练好的（We used IBM’s code and retrained the AMR parser using AMR 3.0 data (LDC2019E81).）。使用的时候要follow原repo给出的installation instructions才能装好，跑的时候要load checkpoint。
- 想要3090跑起来这个parser，依赖怎么安：首先按照AMR-IE的README安装，torch因为要3090，所以去用兼容cuda11以上的版本，他要求1.6不兼容3090，我用1.10也没报错，不过版本不同终究可能会导致复现结果的不同吧。其余三个包用pip安就行，然后安一下lxml和nltk。最重要的是parser相关依赖的安装，不能直接去pip或者conda fairseq，而是要follow原repo给出的installation instructions从本地对fairseq进行patch后再安装，具体follow原repo也就够了。
- 关于生成的数据：我用两台机器都对他们给出的genia2011的oneiejson文件进行parse，跑出来的结果是同样的，这证明用pretrained parser进行amr parse没有随机性，但是这两个结果都与他们放出来的预处理好的文件不一，我想有可能是因为依赖版本不同？这个应该影响不大。

## Datasets
### ACE-2005 and ERE
The ACE-2005 and ERE datasets are only available at the [LDC website](https://catalog.ldc.upenn.edu/LDC2006T06). Please use the following steps for preprocessing and AMR parsing.
+ Use the scripts in [OneIE](http://blender.cs.illinois.edu/software/oneie/) to obtain the data files `train.oneie.json`, `dev.oneie.json`, and `test.oneie.json` for the OneIE model.
+ Use `data/transform_for_amrie.py` to transform the OneIE formatted data files to fit in our model. For example: `python ./data/transform_for_amrie.py -i [INPUT_DATA_DIR] -o [OUTPUT_DATA_DIR]`.
+ Use `./process_amr.py` to conduct AMR parsing and generate the `.pkl` files for training the model.

After preprocessing, each dataset split should contain a `.json` file (containing the IE annotations) and a `.pkl` file (containing AMR information). For example, for the training set of ACE-2005, we have `train.oneie.json` and `train_graphs.pkl`. You can also refer to the publicly available [GENIA datasets](https://drive.google.com/file/d/1tnGyyJo7Enesqv8R1Mpng7c1U5lEzLqm/view?usp=sharing) for more detailed dataset formats in our model.

### GENIA 2011 and 2013
We release the preprocessed data for GENIA 2011 and 2013 along with the AMR graphs for each sentence at [this link](https://drive.google.com/file/d/1tnGyyJo7Enesqv8R1Mpng7c1U5lEzLqm/view?usp=sharing). Please unzip the file and put all the folders into `./data/` before training the models. 

## Training
To train a model, you can run `python train.py -c [CONFIG_FILE] -g [GPU_ID] -n [NAME_OF_THIS_RUN]`. The example config files are provided in `./config/`. For example, to train a model on GENIA 2011 joint IE dataset, you can run:

`python train.py -c config/genia_2011.json -g 0 -n your_preferred_name`

The average training time for a single model would be ~15 hours on Tesla V100 GPUs, with ~10GB usage of GPU memory.

## Acknowledgement
This code is largely derived from [OneIE](http://blender.cs.illinois.edu/software/oneie/). Our great thanks to [Lin et al.](https://www.aclweb.org/anthology/2020.acl-main.713/) and [Astudillo et al.](https://www.aclweb.org/anthology/2020.findings-emnlp.89/) for publicizing their codes for the OneIE model and the AMR parser!

Please contact zixuan11@illinois.edu if you have any questions.
If you use this code as part of your research, please cite the following paper:
```
@inproceedings{amrie2021,
  author    = {Zixuan Zhang and Heng Ji},
  title     = {Abstract Meaning Representation Guided Graph Encoding and Decodingfor Joint Information Extraction},
  booktitle = {Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT)},
  year      = {2021},
  pages     = {39--49},
  url       = {https://www.aclweb.org/anthology/2021.naacl-main.4/}
  }
```
