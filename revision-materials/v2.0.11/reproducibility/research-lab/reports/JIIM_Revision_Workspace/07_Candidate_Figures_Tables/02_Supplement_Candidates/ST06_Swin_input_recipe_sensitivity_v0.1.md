# ST06: Swin input recipe sensitivity

Archived table rendering v0.1.

| Dataset | Paired contexts | Input ΔBA (pp) | 95% paired CI (pp) | Holm p (two pairs) |
| --- | --- | --- | --- | --- |
| ISIC2019 | 15 | 1.960 | [1.165, 3.226] | 0.1250 |
| MURA | 15 | -0.077 | [-0.562, 0.500] | 0.7500 |

Swin weight-evaluation recipe minus shared recipe, on the same 15 contexts per dataset. Existing paired nested intervals and conditional five-split sign-flip tests; the minimum two-sided exact p is 0.0625, and Holm adjusts two dataset comparisons.
