# DeepMap: a deep learning enabled python package for genotype to phenotype mapping in rice breeding
The DeepMap is a deep learning based python package for genotype to phenotype mapping in rice breeding and can be utilize for other prediction-based studies. It is based on multiple genetic-interactions approach for data augmentation and objects for the simplicity/reproducibility of the research by employing four lines of genomic prediction code.

## Installation

### System requirements
* python (>=3.6)
* GPU 1050Ti => (optional)
* C++17 compatible compiler (tested on Apple clang version 12.0.0 and GCC version 7.4.0) (optional)

### Install from Python Package Index (PYPY) 

We provide the python package for reproducing this research. You can download the package and install it as follows:

    % pip install -i https://test.pypi.org/simple/ DeepMap

### Install from sdist

You can build and install from the source distribution downloaded from:

    % git clone https://github.com/IRRISouthAsiaHub/DeepMap

To build DeepMap from the source distribution, you need a C++17 compatible compiler.

## Data Preprocessing

You need to convert your files into appropriate data-format to run the package successfully (Data processing steps are mentioned in [``1.DataPreprocessingScript - R``](https://github.com/ajaykumarirri/DeepMap/tree/main/1.%20DataPreprocessingScript%20-%20R)).


## Training 

The training step is easiest, and requires only four line of code execution mentioned in [``2. Calling gDeepPredict``](https://github.com/ajaykumarirri/DeepMap/tree/main/2.%20Calling%20gDeepPredict)

Please follow the steps mentioned in script and it will give you the results in your current working directory (CWD).

## Web server

A web server is working at https://test.pypi.org/project/DeepMap-1.0/.


## References

* This research work is under-review and you are seeing in private mode. Please do not share the link.
* The DeepMap is a deep learning based python package for genotype to phenotype mapping in rice breeding and can be utilize for other prediction-based studies. It is based on multiple genetic-interactions approach for data augmentation and objects for the simplicity/reproducibility of the research by employing four lines of genomic prediction code. Please contact, p.sinha@irri.org for more information.
