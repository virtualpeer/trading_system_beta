from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
"""
label된 것을 train_set, test_set 나눈 후 train_set에서 Variance를 낮추기 위한 Ensemble하는 세 가지 방법
1. sequential bootstrap을 한 후 N개의 train_set으로 쪼개고 bagging
2. sklearn.ensemble.BaggingClassifier(max_samples=score.mean()) 이용해서 알아서 bagging
3. RF 이용 (1,2를 함께 써서 또는 다른 방법으로 (Chapter 6.4 참고))
"""

def bagging(type, **kwargs):
    assert type in [0, 1, 2]
    if type == 0:
        try:
            return RandomForestClassifier(n_estimators=kwargs[n_estimators], class_weight='balanced_subsample', criterion='entropy')
        except KeyError:
            print('n_estimators needed')
    elif type == 1:
        if not('n_estimators' in kwargs and 'max_samples' in kwargs):
            raise Exception('n_estimators and max_samples must be provided')
        return BaggingClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy', max_features='auto', class_weight='balanced'), **kwargs)

    else:
        if not ('n_estimators' in kwargs and 'max_samples' in kwargs):
            raise Exception('n_estimators and max_samples must be provided')
        return BaggingClassifier(base_estimator=RandomForestClassifier(n_estimators=1, criterion='entropy', bootstrap=False, class_weight='balanced_subsample'), **kwargs)





