'''
Helper functions to apply Type-II 2D DCT over image blocks.

Copyright(c) 2022 Matteo Naccari
All Rights Reserved.

email: matteo.naccari@gmail.com | matteo.naccari@polimi.it | matteo.naccari@lx.it.pt

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
'''


import math
from typing import Any

import numpy as np
from nptyping import NDArray


def compute_dct_matrix(block_size: int) -> NDArray[(Any, Any), np.float64]:
    m1, m2 = np.meshgrid(range(block_size), range(block_size))
    normaliser = np.ones((block_size, block_size), np.float64)
    normaliser[0, ::] = 1.0 / math.sqrt(block_size)
    normaliser[1::, ::] = math.sqrt(2.0 / block_size)
    cosine_basis = np.cos(np.multiply(m2, 2.0 * m1 + 1.0) * np.pi / (2.0 * block_size))
    T = np.multiply(cosine_basis, normaliser)

    return T


def compute_dct(block: NDArray[(Any, Any), np.float64], T: NDArray[(Any, Any), np.float64]) -> NDArray[(Any, Any), np.float64]:
    Tt = np.transpose(T)

    block_t = np.matmul(T, np.matmul(block, Tt))
    return block_t
