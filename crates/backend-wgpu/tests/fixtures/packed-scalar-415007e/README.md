# Scalar packed shader source oracle

These 12 text fragments were copied from clean commit `415007ea32509ff72cd130d8c4de87860f75e74e` before changing the word loader. That commit's backend source matches the frozen native host baseline `bf32d7a`.

The portable source test independently assembles the 8 NF4/ternary single, pair, input-SwiGLU and output-SwiGLU compositions for all 4 staging selections. It compares the resulting bytes with the current registry's assembled source. The fixture's F32 body contains the original scalar load block, so replacing that block and reconstructing the same source can't silently alter another arithmetic or staging line.

Keep these fixtures frozen. An intentional change to those compositions needs a separate review of this compatibility contract.
