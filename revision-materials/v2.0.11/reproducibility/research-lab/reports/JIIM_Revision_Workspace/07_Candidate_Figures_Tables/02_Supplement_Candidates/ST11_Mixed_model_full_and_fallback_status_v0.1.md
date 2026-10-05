# ST11: Mixed model full and fallback status

Candidate v0.1; not yet inserted in the manuscript.

| Dataset / rule | Specification | Converged | Singular | Splits / seeds |
| --- | --- | --- | --- | --- |
| SIPaKMeD / A | sap_full_formula_attempt | True | True | 10 / 5 |
| SIPaKMeD / A | primary_reported_fallback_model | True | False | 10 / 5 |
| SIPaKMeD / B | sap_full_formula_attempt | True | True | 10 / 5 |
| SIPaKMeD / B | primary_reported_fallback_model | True | False | 10 / 5 |
| OrganAMNIST / A | sap_full_formula_attempt | True | True | 10 / 5 |
| OrganAMNIST / A | primary_reported_fallback_model | True | False | 10 / 5 |
| OrganAMNIST / B | sap_full_formula_attempt | True | True | 10 / 5 |
| OrganAMNIST / B | primary_reported_fallback_model | True | False | 10 / 5 |

Existing original full and reported fallback fits. Full models include a seed random intercept and are singular; the reported split and model×split fallback fits are nonsingular. These are diagnostics of the existing fits.
