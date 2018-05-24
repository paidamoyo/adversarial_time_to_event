import numpy as np
import pandas

from utils.pre_processing import one_hot_encoder, formatted_data, one_hot_indices, missing_proportion, \
    get_train_median_mode


# TODO Fix merge on ID
def generate_data():
    # 'PUBCSNUM' 'REG' 'MAR_STAT' 'RACE1V' 'NHIADE' 'SEX' 'AGE_DX' 'YR_BRTH' 'SEQ_NUM' 'MDXRECMP'
    #  'YEAR_DX' 'PRIMSITE' 'LATERAL' 'HISTO2V' 'BEHO2V' 'HISTO3V' 'BEHO3V' 'GRADE' 'DX_CONF'
    # 'REPT_SRC' 'EOD10_SZ' 'EOD10_EX' 'EOD10_PE' 'EOD10_ND' 'EOD10_PN' 'EOD10_NE' 'EOD13' 'EOD2'
    # 'EOD4' 'EOD_CODE' 'TUMOR_1V' 'TUMOR_2V' 'TUMOR_3V' 'CSTUMSIZ' 'CSEXTEN' 'CSLYMPHN' 'CSMETSDX'
    #  'CS1SITE' 'CS2SITE' 'CS3SITE' 'CS4SITE' 'CS5SITE' 'CS6SITE' 'CS25SITE' 'DAJCCT' 'DAJCCN' 'DAJCCM'
    # 'DAJCCSTG' 'DSS1977S' 'DSS2000S' 'DAJCCFL' 'CSVFIRST' 'CSVLATES' 'CSVCURRENT' 'SURGPRIF' 'SURGSCOF'
    # 'SURGSITF' 'NUMNODES' 'NO_SURG' 'SS_SURG' 'SURGSCOP' 'SURGSITE' 'REC_NO' 'TYPE_FU' 'AGE_1REC' 'SITERWHO'
    # 'ICDOTO9V' 'ICDOT10V' 'ICCC3WHO' 'ICCC3XWHO' 'BEHTREND' 'HISTREC' 'HISTRECB' 'CS0204SCHEMA' 'RAC_RECA'
    #  'RAC_RECY' 'ORIGRECB' 'HST_STGA' 'AJCC_STG' 'AJ_3SEER' 'SSS77VZ' 'SSSM2KPZ' 'FIRSTPRM' 'ST_CNTY' 'CODPUB'
    #  'CODPUBKM' 'STAT_REC' 'IHSLINK' 'SUMM2K' 'AYASITERWHO' 'LYMSUBRWHO' 'VSRTSADX' 'ODTHCLASS' 'CSTSEVAL'
    # 'CSRGEVAL' 'CSMTEVAL' 'INTPRIM' 'ERSTATUS' 'PRSTATUS' 'CSSCHEMA' 'CS8SITE' 'CS10SITE' 'CS11SITE' 'CS13SITE'
    # 'CS15SITE' 'CS16SITE' 'VASINV' 'SRV_TIME_MON' 'SRV_TIME_MON_FLAG' 'INSREC_PUB' 'DAJCC7T' 'DAJCC7N' 'DAJCC7M'
    # 'DAJCC7STG' 'ADJTM_6VALUE' 'ADJNM_6VALUE' 'ADJM_6VALUE' 'ADJAJCCSTG' 'CS7SITE' 'CS9SITE' 'CS12SITE' 'HER2'
    # 'BRST_SUB' 'ANNARBOR' 'CSMETSDXB_PUB' 'CSMETSDXBR_PUB' 'CSMETSDXLIV_PUB'
    # 'CSMETSDXLUNG_PUB' 'T_VALUE' 'N_VALUE' 'M_VALUE' 'MALIGCOUNT' 'BENBORDCOUNT'
    # Load DATA
    # TODO STATE-COUNTY RECODE (ST_CNTY)# variable was dropped and replaced with the elevation , lat , and lng variables
    # TODO for all three datasets as illustrated in Table

    # TODO  na_median: $CS8SITE, $CS10SITE, $CS11SITE, $CS13SITE, $CS15SITE, $CS16SITE, $VASINV, $DAJCC7T,
    # TODO na_median: DAJCC7N, DAJCC7M, DAJCC7STG, $CS7SITE , $CS9SITE, $CS12SITE, $CSMETSDXB_PUB,
    # TODO na_median:  $CSMETSDXBR_PUB, $CSMETSDXLIV_PUB, $CSMETSDXLUNG_PUB

    # TODO: Review uncoded list uncoded:['CSTUMSIZ', 'M_VALUE', 'T_VALUE', 'CSRGEVAL', 'CSMTEVAL', 'EOD10_EX',
    # TODO: uncoded 'NUMNODES', 'CSTSEVAL', 'EOD10_SZ', 'AGE_DX', 'EOD10_NE', 'EOD10_ND', 'MALIGCOUNT', 'BENBORDCOUNT',
    # TODO: uncoded 'TYPE_FU', 'N_VALUE', 'EOD10_PN', 'EOD_CODE']

    # Covariates groupings
    # identifiers = ['PUBCSNUM', 'STAT_REC', 'ST_CNTY']
    identifiers = ['STAT_REC', 'ST_CNTY']

    #
    one_hot_encode_list = ['PRIMSITE', 'ICDOT10V', 'YEAR_DX', 'REG', 'MAR_STAT', 'RACE1V',
                           'NHIADE', 'MDXRECMP', 'LATERAL', 'HISTO2V', 'BEHO2V', 'HISTO3V', 'BEHO3V', 'GRADE',
                           'DX_CONF', 'REPT_SRC', 'TUMOR_1V', 'TUMOR_2V', 'TUMOR_3V', 'CSEXTEN', 'CSLYMPHN', 'CSMETSDX',
                           'CS1SITE', 'CS2SITE', 'CS3SITE', 'CS4SITE', 'CS5SITE', 'CS6SITE', 'CS25SITE', 'DAJCCT',
                           'DAJCCN', 'DAJCCM', 'DAJCCSTG', 'DSS1977S', 'DSS2000S', 'DAJCCFL', 'CSVFIRST', 'CSVLATES',
                           'CSVCURRENT', 'SURGPRIF', 'SURGSCOF', 'SURGSITF', 'NO_SURG', 'SS_SURG', 'SURGSCOP',
                           'SURGSITE', 'AGE_1REC', 'ICDOTO9V', 'ICCC3WHO', 'ICCC3XWHO', 'BEHTREND', 'HISTREC',
                           'HISTRECB', 'CS0204SCHEMA', 'RAC_RECA', 'RAC_RECY', 'ORIGRECB', 'HST_STGA', 'AJCC_STG',
                           'AJ_3SEER', 'SSS77VZ', 'SSSM2KPZ', 'FIRSTPRM', 'IHSLINK', 'SUMM2K', 'AYASITERWHO',
                           'LYMSUBRWHO', 'INTPRIM', 'ERSTATUS', 'PRSTATUS', 'CSSCHEMA', 'INSREC_PUB', 'ADJTM_6VALUE',
                           'ADJNM_6VALUE', 'ADJM_6VALUE', 'ADJAJCCSTG', 'HER2', 'BRST_SUB', 'ANNARBOR']

    redudant_variables = ['YR_BRTH', 'SEX', 'SEQ_NUM', 'REC_NO', 'SITERWHO', 'TYPE_FU']

    outcomes = ['SRV_TIME_MON', 'STAT_REC', 'CODPUB', 'CODPUBKM', 'VSRTSADX', 'ODTHCLASS', 'SRV_TIME_MON_FLAG']

    na_median = ['EOD10_PE', 'EOD13', 'EOD2', 'EOD4', 'CS8SITE', 'CS10SITE', 'CS11SITE',
                 'CS13SITE', 'CS15SITE', 'CS16SITE', 'VASINV', 'DAJCC7T', 'DAJCC7N', 'DAJCC7M', 'DAJCC7STG', 'CS7SITE',
                 'CS9SITE', 'CS12SITE', 'CSMETSDXB_PUB', 'CSMETSDXBR_PUB', 'CSMETSDXLIV_PUB', 'CSMETSDXLUNG_PUB']

    to_drop = identifiers + outcomes + redudant_variables + na_median
    print("dropped_columns:{}".format(len(to_drop)))
    print("size_categorical:{}".format(len(one_hot_encode_list)))

    #  SEX ,  MARITAL STATUS AT DX , RACE/ETHNICITY ,  SPANISH/HISPANIC ORIGIN ,  GRADE ,
    #  PRIMARY SITE , 13/28  LATERALITY ,  SEER HISTORIC STAGE A ,  HISTOLOGY RECODE--BROAD GROUPINGS ,
    # MONTH OF DIAGNOSIS ,  VITAL STATUS RECODE , and the STATE-COUNTY RECODE
    # variable was dropped and replaced with the elevation , lat , and lng variables
    # for all three datasets as illustrated in Table
    # TODO cause specific: VSRTSADX: alive or dead other cause =0, ODTHCLASS: alived or dead of cancer=0
    #  TODO: TYPE_FU may be include Sanfranciso
    np.random.seed(31415)
    data_path = '/data/ash/seer/'
    print("data_path:{}".format(data_path))
    data_frame = pandas.read_csv(data_path + 'seers_data.csv', index_col=0)
    print("all_data:{}".format(data_frame.shape))
    data_frame = select_cohort_data(data_frame)

    # breast= 26000, cvd = 50060, alive = 1, dead = 4
    cancer_deaths = data_frame[np.logical_and(data_frame.CODPUB == 26000, data_frame.STAT_REC == 4)]
    cvd_deaths = data_frame[np.logical_and(data_frame.CODPUB == 50060, data_frame.STAT_REC == 4)]
    all_deaths = data_frame[data_frame.STAT_REC == 4]

    size = data_frame.shape[0]
    cancer = cancer_deaths.shape[0]
    cvd = cvd_deaths.shape[0]
    deaths = all_deaths.shape[0]
    print("cancer_deaths:{}, cvd_deaths:{}, all_deaths:{},  other_deaths:{}".format(cancer / size,
                                                                                    cvd / size,
                                                                                    deaths / size, (deaths - cancer
                                                                                                    - cvd) / size))

    print("head of data:{}, data shape:{}".format(data_frame.head(), data_frame.shape))

    all_columns = list(data_frame.columns.values)
    print("all_columns:{}".format(all_columns))
    uncoded_covariates = list(set(all_columns) ^ set(one_hot_encode_list + to_drop))
    print("uncoded_covariates:{}{}".format(uncoded_covariates, len(uncoded_covariates)))

    print("missing:{}".format(missing_proportion(data_frame.drop(labels=to_drop, axis=1))))
    # Replace missing values with median or mode
    # data_frame = replace_na_with_mode(data=data_frame, replace_ls=one_hot_encode_list)

    # data_frame = replace_na_with_median(data=data_frame, replace_ls=replace_ls)
    print("columns with missing values:{}".format(data_frame.columns[data_frame.isnull().any()]))

    # One hot Encoding
    data_frame = one_hot_encoder(data=data_frame, encode=one_hot_encode_list)
    print("columns with missing values:{}".format(data_frame.columns[data_frame.isnull().any()]))

    # Split into test train and valid
    t_data = data_frame[['SRV_TIME_MON']]
    # STAT_REC==4 death 1 = alive
    e_data = data_frame[['STAT_REC']]
    outcome_data = data_frame[['CODPUB']]
    x_data = data_frame.drop(labels=to_drop, axis=1)
    encoded_indices = one_hot_indices(x_data, one_hot_encode_list)
    print("head of x data:{}, data shape:{}".format(x_data.head(), x_data.shape))
    print("columns with missing values:{}".format(x_data.columns[x_data.isnull().any()]))
    covariates = np.array(x_data.columns.values)
    np.save(file=('%sdata_covariates' % data_path), arr=covariates)
    x = np.array(x_data).reshape(x_data.shape)
    t = np.array(t_data).reshape(len(t_data))
    # Transform code to 1 = dead 0 = alive,   # STAT_REC==4 death 1 = alive
    e = np.array(e_data).reshape(len(e_data))
    alive = e == 1
    e[alive] = 0
    e[np.logical_not(alive)] = 1

    # Type of outcome breast= 26000, cvd = 50060, alive = 1, dead = 4 (0=alive, 1=cancer, 2=cvd, 3=other)
    outcomes = np.array(outcome_data).reshape(len(outcome_data))
    print(outcomes[0])
    death = e == 1
    cancer = outcomes == 26000
    cvd = outcomes == 50060
    other = np.logical_and(np.logical_not(cancer), np.logical_not(cvd))
    encoded_cancer = np.zeros(shape=len(e))
    encoded_cvd = np.zeros(shape=len(e))
    encoded_other = np.zeros(shape=len(e))

    encoded_cancer[np.logical_and(cancer, death)] = 1
    encoded_cvd[np.logical_and(cvd, death)] = 1
    encoded_other[np.logical_and(other, death)] = 1

    print("cause of death:cancer:{}, cvd:{}, other:{}".format(sum(encoded_cancer / len(t)), sum(encoded_cvd) / len(t),
                                                              sum(encoded_other) / len(t)))
    # assert_nan([x, t, e, encoded_other, encoded_cvd, encoded_cancer])

    idx = np.arange(0, x.shape[0])
    print("x_shape:{}".format(x.shape))

    np.random.shuffle(idx)
    x = x[idx]
    t = t[idx]
    e = e[idx]
    encoded_cancer = encoded_cancer[idx]
    encoded_cvd = encoded_cvd[idx]
    encoded_other = encoded_other[idx]
    end_time = max(t)
    print("end_time:{}".format(end_time))
    print("observed percent:{}".format(sum(e) / len(e)))
    print("shuffled x:{}, t:{}, e:{}, len:{}".format(x[0], t[0], e[0], len(t)))
    num_examples = int(0.80 * len(e))
    print("num_examples:{}".format(num_examples))
    train_idx = idx[0: num_examples]
    split = int((len(t) - num_examples) / 2)

    test_idx = idx[num_examples: num_examples + split]
    valid_idx = idx[num_examples + split: len(t)]
    print("test:{}, valid:{}, train:{}, all: {}".format(len(test_idx), len(valid_idx), num_examples,
                                                        len(test_idx) + len(valid_idx) + num_examples))
    imputation_values = get_train_median_mode(x=np.array(x[train_idx]), categorial=encoded_indices)
    print("imputation_values:{}".format(imputation_values))
    dataset = {
        'train': outcome_formatted_data(x=x, t=t, e=e, idx=train_idx, name='Train',
                                        data_type='seer',
                                        path=data_path,
                                        outcomes={'cancer': encoded_cancer[train_idx], 'cvd': encoded_cvd[train_idx],
                                                  'other': encoded_other[train_idx]}),
        'test': outcome_formatted_data(x=x, t=t, e=e, idx=test_idx, name='Test',
                                       data_type='seer',
                                       path=data_path,
                                       outcomes={'cancer': encoded_cancer[test_idx], 'cvd': encoded_cvd[test_idx],
                                                 'other': encoded_other[test_idx]}),
        'valid': outcome_formatted_data(x=x, t=t, e=e, idx=valid_idx, name='Valid',
                                        data_type='seer',
                                        path=data_path,
                                        outcomes={'cancer': encoded_cancer[valid_idx], 'cvd': encoded_cvd[valid_idx],
                                                  'other': encoded_other[valid_idx]}),
        'end_t': end_time,
        'covariates': covariates,
        'one_hot_indices': encoded_indices,
        # 'one_hot_indices': [],
        'imputation_values': imputation_values
    }
    return dataset


def outcome_formatted_data(x, t, e, idx, name, data_type, path, outcomes):
    survival_data = formatted_data(x=x, t=t, e=e, idx=idx)
    cancer = outcomes['cancer']
    cvd = outcomes['cvd']
    other = outcomes['other']
    np.savetxt('{}{}_cancer_outcome_{}'.format(path, data_type, name), cancer)
    np.savetxt('{}{}_cvd_outcome_{}'.format(path, data_type, name), cvd)
    np.savetxt('{}{}_other_outcome_{}'.format(path, data_type, name), other)
    survival_data.update({'outcomes': outcomes})

    print("cancer:{}, cvd:{}, other:{}".format(cancer.shape, cvd.shape, other.shape))
    cancer = np.expand_dims(cancer, axis=1)
    cvd = np.expand_dims(cvd, axis=1)
    other = np.expand_dims(other, axis=1)
    all_data = np.concatenate((cancer, cvd, other), axis=1)
    binary_outcomes = pandas.DataFrame(all_data, columns=['cancer', 'cvd', 'other'])
    binary_outcomes.to_csv('{}{}_{}_binary_outcomes'.format(path, data_type, name), encoding='utf-8', index=False)
    return survival_data


def select_cohort_data(data):
    # Select  1992-2007
    cohort_year = np.arange(2007 - 1992 + 1) + 1992
    print("cohort_year:{}".format(cohort_year))
    data = data[data.YEAR_DX.isin(cohort_year)]
    print("cohort data:{}".format(data.shape))

    # Only (12* 10=120 months)10 year follow up, completely observerd survival time and follow up
    data = data[data.SRV_TIME_MON <= 120]
    print("observed 10 year cohort:{}".format(data.shape))

    # Only active follow up
    data = data[data.TYPE_FU == 2]
    print("only active follow up data:{}".format(data.shape))

    # Only complete time
    data = data[data.SRV_TIME_MON_FLAG == 1]
    print("observed survival time:{}".format(data.shape))

    # TODO compare SRV_TIME_MON != 9999 and data_frame.SRV_TIME_MON_FLAG == 1
    # Only Known time
    data = data[data.SRV_TIME_MON != 9999]
    print("Known survival time {}".format(data.shape))

    # Remove duplicate ids only fist sequence
    data = data[data.SEQ_NUM == 0]
    print("Removed dublicate patients:{}".format(data.shape))

    # Remove unknown ER and PR status
    data = data[data.ERSTATUS != 4]
    print("Known ER status data:{}".format(data.shape))

    data = data[data.PRSTATUS != 4]
    print("Known PR status  data:{}".format(data.shape))

    # Only Microscopically Confirmed
    data = data[data.DX_CONF != 9]
    print("Microscopically confirmed data:{}".format(data.shape))

    # age greater than 20 and not unkown
    data = data[data.AGE_DX > 20]
    print("age greater than 20: {}".format(data.shape))
    data = data[data.AGE_DX != 999]
    print("age is known: {}".format(data.shape))

    # Female only
    data = data[data.SEX == 2]
    print("Female only: {}".format(data.shape))

    # Reporting source not Autopys-6 or death certificate-7
    data = data[data.REPT_SRC.isin([1, 2, 3, 4, 5, 8])]
    print("Reliable reporting source: {}".format(data.shape))

    # Removed unstaged
    data = data[data.HST_STGA != 9]
    print("Removed unstaged: {}".format(data.shape))

    # Remove ungraded
    data = data[data.GRADE != 9]
    print("Removed ungraded: {}".format(data.shape))
    return data
