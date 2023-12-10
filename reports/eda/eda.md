# EDA

## Insights
- Missing values are already encoded as some categories (e.g. for `PAY_REC`, it is 9=Unknown) $\implies$ encode them back as missing values for appropriate imputation.
- Some values are capped at different thresholds $\implies$ introduce binary flags for whether clipped or not?


## Variables, see [this guide](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/DVS/natality/UserGuide2018-508.pdf)

**Numerical variables**
- `BMI` - *Body Mass Index* - **NA=99.9**
- `WTGAIN` - *Weight Gain* - **NA=99** - **capped at 98**
- `FAGECOMB` - *Father’s Combined Age* - **NA=99** - **capped at 98**
- `PREVIS` - *Number of Prenatal Visits* - **NA=99** - **capped at 98**
- `MAGER` - *Mother’s Single Years of Age* - **no missing** - **capped at 50**
- `CIG_0` - *Cigarettes Before Pregnancy* - **NA=99** - **capped at 98**
- `M_Ht_In` - *Mother’s Height in Total Inches* - **NA=99**
- `PRIORDEAD` - *Prior Births Now Dead* - **NA=99**
- `PRIORLIVE` - *Prior Births Now Living* - **NA=99**
- `PRIORTERM` - *Prior Other Terminations* - **NA=99**
- `PRECARE` - *Month Prenatal Care Began* - **NA=99**
- `RF_CESARN` - *Number of Previous Cesareans* - **NA=99**
- `PWgt_R` - *Pre-pregnancy Weight Recode* - **NA=999**

**Categorical variables**
- `LD_INDL` - *Induction of Labor* - **no missing**
- `RF_CESAR` - *Previous Cesarean* - **no missing**
- `SEX` - *Sex of Infant* - **no missing**
- `DMAR` - *Marital Status* - **NA=' '**

**Numerical variables that should be converted to categorical**
- `MEDUC` - *Mother’s Education* - **NA=9**
- `FEDUC` - *Father’s Education* - **NA=9**
- `PAY` - *Payment Source for Delivery* - **NA=9**
- `BFACIL` - *Birth Place* - **NA=9**
- `ATTEND` - *Attendant at Birth* - **NA=9**
- `RDMETH_REC` - *Delivery Method Recode* - **NA=9**
- `PAY_REC` - *Payment Recode* - **NA=9**
- `RESTATUS` - *Residence Status* - **no missing**
- `NO_INFEC` - *No Infections Reported* - **NA=9**
- `MBSTATE_REC` - *Mother’s Nativity* - **NA=3**
- `NO_RISKS` - *No Risk Factors Reported* - **NA=9**
- `NO_MMORB` - *No Maternal Morbidity Reported* - **NA=9**


**Numerical variables that are cyclical**
- `DLMP_MM` - *Last Normal Menses Month* - **NA=99** - Combine with `DOB_MM` to form numerical number of months between last normal menses and birth month?
- `DOB_MM` - *Birth Month* - **no missing**
- `DOB_TT` - *Time of Birth* - **NA=9999**
- `DOB_WK` - *Birth Day of Week* - **no missing**


### Requires further treatment (e.g. split up)
- `ILLB_R` - *Interval Since Last Live Birth Recode* - Effectively encodes different things (000-003 Plural delivery, 004-300 Months since last live birth, 888 Not applicable / 1st live birth, 999 Unknown or not stated)
- `ILOP_R` - *Interval Since Last Other Pregnancy Recode* - Effectively encodes different things (000-003 Plural delivery, 004-300 Months since last live birth, 888 Not applicable / 1st natality event, 999 Unknown or not stated)
- `ILP_R` - *Interval Since Last Other Pregnancy Recode* - Effectively encodes different things (000-003 Plural delivery, 004-300 Months since last live birth, 888 Not applicable / no previous pregnancy, 999 Unknown or not stated)


**For inductive bias reasons**

**Target**
- `DBWT` - *Birth Weight (Detail in Grams)*
