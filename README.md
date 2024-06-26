# Video Coding Tutorial (VCT)
Tutorials on the fundamentals tools used in image and video compression.

## Introduction
The Video Coding Tutorial (VCT) repository provides a set of Jupyter notebooks to explain the foundations of image and data compression with focus on the following aspects:
 * [Entropy coding](./entropy-coding)
 * [Transform coding](./transform-coding)
 * [Quantisation](./quantisation)
 * [Image codecs](./image-codecs/)

The material has been prepared with the aim to complement the theoretical foundations that can be learned from some of the classical text books on image and video compression (see below for a list of references). It is believed that seeing how the theory is put into limited, yet practical examples, readers will strengthen their understanding in the area of compression of visual data. Accordingly, along with a description and explanation on the coding tools/algorithms/methods presented, a Python implementation is also provided so that the reader can perform additional code debugging, investigations, etc. to expand their understanding in the area of image/video compression. As of April 2024 it is known to the author that some key aspects of video codecs such as inter frame prediction are still not present in this tutorial. Please bear with me: being living on planet Earth where days are only of 24 hours and needing to work to find the means the support myself, I still haven't found time to get round these key aspects of video coding. I will get there, eventually.

## References
Some good textbooks that can be used to learn the fundamentals as well as being up to date with the state-of-the-art are:
 * N. S. Jayant and P. Noll., "Digital coding of waveforms", Prentice Hall, 688 pages, 1984.
 * A. N. Netravali, B. G. Haskell, "Digital pictures - Representation, compression, and standards", 2nd edition, Plenum Press, 686 pages, 1995.
 * D. S. Taubman, M. W. Marcellin, "JPEG 2000 - Image compression fundamentals, standards and practice", Kluwer Academic Press, 773 pages, 2002.
 * D. Bull, "Communicating pictures: A course in image and video coding", Associate Press, 560 pages, 2014.

## Installation
The `requirements.txt` file lists all the pacakges required to use the tutorials. If using `pip`, just type the following in a command terminal window:
```bash
> pip install -r requirements.txt
```

## License
The copyright in this collection of software modules is being made available under the BSD
License, included below. This software may be subject to other third party
and contributor rights, including patent rights, and no such rights are
granted under this license.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
 * Neither the name of the author may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES LOSS OF USE, DATA, OR PROFITS OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.

## Copyright
Copyright(c) 2024 Matteo Naccari - All Rights Reserved.

## Contributing
Feel free to submit pull requests and issue tickets with suggestions to improve, fix and expand this collection of tutorials.