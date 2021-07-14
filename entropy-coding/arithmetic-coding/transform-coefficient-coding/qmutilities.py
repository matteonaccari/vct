'''
Probability state machine for the QM codec. More details available in Section
D.1.5.1 of: https://www.itu.int/rec/T-REC-T.81-199209-I/en

Each entry in the list represents the following:
    - qe value, i.e. probability of the LPS
    - next index in case an LPS is processed
    - next index in case an MPS is processed
    - LPS and MPS switch

Copyright(c) 2021 Matteo Naccari
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

from dataclasses import dataclass

endOfArithmeticCodingMarker = 0xD3

probability_state_machine = [
    [0x5A1D, 1, 1, True], [0x2586, 14, 2, False], [
        0x1114, 16, 3, False], [0x80B, 18, 4, False],
    [0x3D8, 20, 5, False], [0x1DA, 23, 6, False], [
        0xE5, 25, 7, False], [0x6F, 28, 8, False],
    [0x36, 30, 9, False], [0x1A, 33, 10, False], [
        0xD, 35, 11, False], [0x6, 9, 12, False],
    [0x3, 10, 13, False], [0x1, 12, 13, False], [
        0x5A7F, 15, 15, True], [0x3F25, 36, 16, False],
    [0x2CF2, 38, 17, False], [0x207C, 39, 18, False], [
        0x17B9, 40, 19, False], [0x1182, 42, 20, False],
    [0xCEF, 43, 21, False], [0x9A1, 45, 22, False], [
        0x72F, 46, 23, False], [0x55C, 48, 24, False],
    [0x406, 49, 25, False], [0x303, 51, 26, False], [
        0x240, 52, 27, False], [0x1B1, 54, 28, False],
    [0x144, 56, 29, False], [0xF5, 57, 30, False], [
        0xB7, 59, 31, False], [0x8A, 60, 32, False],
    [0x68, 62, 33, False], [0x4E, 63, 34, False], [
        0x3B, 32, 35, False], [0x2C, 33, 9, False],
    [0x5AE1, 37, 37, True], [0x484C, 64, 38, False], [
        0x3A0D, 65, 39, False], [0x2EF1, 67, 40, False],
    [0x261F, 68, 41, False], [0x1F33, 69, 42, False], [
        0x19A8, 70, 43, False], [0x1518, 72, 44, False],
    [0x1177, 73, 45, False], [0xE74, 74, 46, False], [
        0xBFB, 75, 47, False], [0x9F8, 77, 48, False],
    [0x861, 78, 49, False], [0x706, 79, 50, False], [
        0x5CD, 48, 51, False], [0x4DE, 50, 52, False],
    [0x40F, 50, 53, False], [0x363, 51, 54, False], [
        0x2D4, 52, 55, False], [0x25C, 53, 56, False],
    [0x1F8, 54, 57, False], [0x1A4, 55, 58, False], [
        0x160, 56, 59, False], [0x125, 57, 60, False],
    [0xF6, 58, 61, False], [0xCB, 59, 62, False], [
        0xAB, 61, 63, False], [0x8F, 61, 2, False],
    [0x5B12, 65, 65, True], [0x4D04, 80, 66, False], [
        0x412C, 81, 67, False], [0x37D8, 82, 68, False],
    [0x2FE8, 83, 69, False], [0x293C, 84, 70, False], [
        0x2379, 86, 71, False], [0x1EDF, 87, 72, False],
    [0x1AA9, 87, 73, False], [0x174E, 72, 74, False], [
        0x1424, 72, 75, False], [0x119C, 74, 76, False],
    [0xF6B, 74, 77, False], [0xD51, 75, 78, False], [
        0xBB6, 77, 79, False], [0xA40, 77, 48, False],
    [0x5832, 80, 81, True], [0x4D1C, 88, 82, False], [
        0x438E, 89, 83, False], [0x3BDD, 90, 84, False],
    [0x34EE, 91, 85, False], [0x2EAE, 92, 86, False], [
        0x299A, 93, 87, False], [0x2516, 86, 71, False],
    [0x5570, 88, 89, True], [0x4CA9, 95, 90, False], [
        0x44D9, 96, 91, False], [0x3E22, 97, 92, False],
    [0x3824, 99, 93, False], [0x32B4, 99, 94, False], [
        0x2E17, 93, 86, False], [0x56A8, 95, 96, True],
    [0x4F46, 101, 97, False], [0x47E5, 102, 98, False], [
        0x41CF, 103, 99, False], [0x3C3D, 104, 100, False],
    [0x375E, 99, 93, False], [0x5231, 105, 102, False], [
        0x4C0F, 106, 103, False], [0x4639, 107, 104, False],
    [0x415E, 103, 99, False], [0x5627, 105, 106, True], [
        0x50E7, 108, 107, False], [0x4B85, 109, 103, False],
    [0x5597, 110, 109, False], [0x504F, 111, 107, False], [
        0x5A10, 110, 111, True], [0x5522, 112, 109, False],
    [0x59EB, 112, 111, True], [0x5A1D, 113, 113, False]
]


@dataclass
class Context:
    idx: int  # Index on the probability state machine
    mps: int  # Semantics for the MPS
