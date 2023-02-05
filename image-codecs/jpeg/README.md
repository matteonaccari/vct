# The JPEG compression standard
This directory contains a set of tutorials to learn the details associated with the JPEG compression standard, a format initially standardised in 1992 which is still widely used nowadays and deployed in the vast majority of hardware and software multimedia devices and platforms.

Our main goal is to provide the reader with a detailed explanation of the main coding tools and motivations for the design choices made, along with a simple Python implementation so that one can understand how the processing works in practice and expand their knowledge from there. Three tutorials are provided, touching the basic components of the standard and then exploring how the coding efficiency can be boosted by leveraging on the support provided to make the encoding more content adaptive. In particular, the following Jupyter notebooks tackle these aspects:

 * [Baseline, sequential DCT processing of the standard](./jpeg-baseline.ipynb)
 * [Design of ad-hoc Huffman tables to improve the entropy coding efficiency](./jpeg-optimised-huffman.ipynb)
 * [Rate distortion optimised quantisation](./jpeg-rdoq.ipynb)

Besides describing and implementing the standard's coding tools, some experiments/tests are provided to show the capabilities of the codec implemented.