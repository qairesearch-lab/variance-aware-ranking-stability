# RT04: Analysis stratum inventory

Archived table rendering v0.1.

| Stratum | Split×seed | Models | Contexts | Runs in reused stratum | Role |
| --- | --- | --- | --- | --- | --- |
| organamnist_A_cnn4_all5 | 10×5 | 4 | 50 | 200 | primary_original |
| organamnist_A_cnn4_common3 | 10×3 | 4 | 30 | 120 | common_seed_sensitivity |
| organamnist_B_cnn4_all5 | 10×5 | 4 | 50 | 200 | primary_original |
| organamnist_B_cnn4_common3 | 10×3 | 4 | 30 | 120 | common_seed_sensitivity |
| sipakmed_A_cnn4_all5 | 10×5 | 4 | 50 | 200 | primary_original |
| sipakmed_A_cnn4_common3 | 10×3 | 4 | 30 | 120 | common_seed_sensitivity |
| sipakmed_B_cnn4_all5 | 10×5 | 4 | 50 | 200 | primary_original |
| sipakmed_B_cnn4_common3 | 10×3 | 4 | 30 | 120 | common_seed_sensitivity |
| isic2019_A_cnn4_all3 | 10×3 | 4 | 30 | 120 | primary_extension |
| isic2019_A_cnn4_pool_contexts | 5×3 | 4 | 15 | 60 | candidate_pool_control |
| isic2019_A_pool5_shared | 5×3 | 5 | 15 | 75 | candidate_pool_sensitivity |
| isic2019_A_pool5_swin_weight_eval | 5×3 | 5 | 15 | 75 | candidate_pool_sensitivity |
| mura_A_cnn4_all3 | 10×3 | 4 | 30 | 120 | primary_extension |
| mura_A_cnn4_pool_contexts | 5×3 | 4 | 15 | 60 | candidate_pool_control |
| mura_A_pool5_shared | 5×3 | 5 | 15 | 75 | candidate_pool_sensitivity |
| mura_A_pool5_swin_weight_eval | 5×3 | 5 | 15 | 75 | candidate_pool_sensitivity |

Original completed primary matrix: 800 runs. Extension matrix: 300 runs. The 16 analysis strata overlap; their context/run counts are not additive independent sample sizes.
