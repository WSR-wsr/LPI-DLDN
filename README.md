# LPI-DLDN
lncRNA-protein interaction prediction method: LPI-DLDN

# Data
Data is available at [NONCODE](http://www.noncode.org/), [NPInter](http://bigdata.ibp.ac.cn/npinter3/index.htm), and [PlncRNADB](http://bis.zju.edu.cn/PlncRNADB/).

# Feature selection
LncRNA feature: [PyFeat](https://github.com/mrzResearchArena/PyFeat)

Protein feature: [Biotriangle](http://biotriangle.scbdd.com/protein/index/)

# Environment
* python-3.7.9
* numpy-1.19.2
* tensorflow-2.1.0
* pandas-1.1.3
* scikit-learn-0.23.2

# Usage

To run the model, default 5 fold cross validation
```
python LPI-DLDN.py
```

For code questions, please contact wsrgyw@163.com
