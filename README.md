# Refill interruption and return to care among key populations on antiretroviral therapy in Nigeria

Analysis code for the manuscript "Refill Interruption and Return to Care Among Key Populations on Antiretroviral Therapy in Nigeria: A Two-Year Cohort Analysis of Pharmacy Records".

## Contents

- `analysis script.py`: the full analysis. It reads the program's pharmacy, patient and client verification tables and two treatment line lists, and writes every number, table and figure used in the manuscript.

## How to run

```
python "analysis script.py" <pharmacy table (.csv)> <patient table (.csv)> <client verification table (.csv)> <line list, 8 November 2024 (.xlsx)> <line list, 10 April 2026 (.xlsx)> <output folder>
```

The script writes an analysis log, a results workbook and Figure 1 to the output folder.

Requirements: Python 3.13, pandas 2.3, numpy, lifelines 0.30, matplotlib, openpyxl.

## Data

The data are not included. They belong to the program and contain sensitive information on key populations. De-identified data may be made available by the corresponding author on reasonable request and with the permission of the program.

## Reporting

The study is reported according to STROBE (cohort studies) and its RECORD extension.
