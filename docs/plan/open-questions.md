# Open questions

These are intentionally not locked yet.

1. What BPE vocabulary size provides acceptable English/Japanese coverage without unnecessary embedding cost?
2. Which normalization and pre-tokenization rules preserve Japanese text, punctuation, and mixed-script input?
3. What is the first Transformer parameter budget and context length that can be tested comfortably on the target laptop?
4. Which optimizer and schedule should the first tiny training run use?
5. What data licensing and provenance rules will govern every training source?
6. What exact NCM paper/reference implementation settings can be reproduced, and which parts are feasible on CPU?
7. How do Burn Flex and Burn CubeCL CPU compare on the target hardware for forward, backward, and optimizer workloads?
