from typing import Counter
from imblearn.over_sampling import BorderlineSMOTE, SMOTENC, RandomOverSampler, SMOTE, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler


def balance_classes(training_features, training_labels):
    # columns = ['age', 'degree-of-diffe', 'sex_2', 'histologic-type_2', 'histologic-type_3', 'bone_2', 'bone-marrow_2',
    #            'lung_2', 'pleura_2', 'peritoneum_2', 'liver_2', 'brain_2', 'skin_2', 'neck_2', 'supraclavicular_2',
    #            'axillar_2', 'mediastinum_2', 'abdominal_2']
    # sm = SMOTENC(sampling_strategy='minority', categorical_features=columns)  # You need to reduce k or increase the number of instances for the least represented class.
    # sm = BorderlineSMOTE()
    sm = RandomUnderSampler()
    # sm = SMOTE(k_neighbors=3) # Supports multi-class resampling
    # sm = SVMSMOTE() #BorderlineSMOTE, SMOTE, SVMSMOTE - supports multiclass by ovr scheme
    # tomeks = TomekLinks(sampling_strategy='majority')
    # # X_resampled, y_resampled = BorderlineSMOTE().fit_resample(X_resampled, y_resampled)

    # steps = [('smote', sm), ('tomek', tomeks)]
    # pipeline = Pipeline(steps=steps)

    X_resampled, y_resampled = sm.fit_sample(training_features, training_labels)
    # X_resampled, y_resampled = tomeks.fit_sample(X_resampled, y_resampled)
    print('SMOTE {}'.format(Counter(y_resampled)))
    X_resampled.to_csv('../resources/datasets/X_resampled.csv', index=False)
    y_resampled.to_csv('../resources/datasets/y_resampled.csv', header=['binaryClass'], index=False)
    return X_resampled, y_resampled
