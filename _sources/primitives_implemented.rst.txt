Primitives implemented
======================

.. autosummary::

   cascada.primitives.aes
   cascada.primitives.cham
   cascada.primitives.chaskey
   cascada.primitives.feal
   cascada.primitives.hight
   cascada.primitives.lea
   cascada.primitives.multi2
   cascada.primitives.noekeon
   cascada.primitives.picipher
   cascada.primitives.rectangle
   cascada.primitives.shacal1
   cascada.primitives.shacal2
   cascada.primitives.simeck
   cascada.primitives.simon
   cascada.primitives.speck
   cascada.primitives.skinny64
   cascada.primitives.skinny128
   cascada.primitives.tea
   cascada.primitives.xtea


Apart from the suite of tests included in CASCADA,
we have tested the search for low-weight characteristics in the primitives listed above
by revisiting the following work listing the weights of optimal characteristics covering small number of rounds:

 - XOR differential and linear characteristics of AES [Mou+11, Table4]
 - XOR differential and linear characteristics of CHAM, [Roh+19, Table 10, Table11]
 - linear characteristics of Chaskey [LWR16, Table 4]
 - XOR differential and linear characteristics of HIGHT [Yin+17, Table 4, Table 7],
 - XOR [KLT15, Table1], related-key XOR [Wan+18, Table 5] and RX [Lu+20, Table 5] differential characteristics of Simeck,
 - XOR [LLW17, Table 1] related-key XOR [Wan+18, Table 5] and RX [Lu+20, Table 5] differential characteristics of Simon,
 - XOR differential and linear characteristics of SKINNY [Bei+16, Table 7]
 - XOR differential [BVC16, Table 4] and linear [LWR16, Table 2] characteristics of Speck
 - XOR differential and linear characteristics of RECTANGLE [https://eprint.iacr.org/2019/019, Table 4]

You can find the references of the form ``NameYear`` in the paper
`CASCADA: Characteristic Automated Search of Cryptographic Algorithms for Distinguishing Attacks <https://eprint.iacr.org/2022/513>`_.