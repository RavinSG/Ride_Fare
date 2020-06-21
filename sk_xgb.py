from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
from numpy import savetxt
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from bayes_opt import BayesianOptimization
from matplotlib import pyplot as plt
from xgboost import XGBClassifier, plot_importance

from sklearn.model_selection import train_test_split
from update_search_space import update_values

RANDOM_SEED = 6


def get_distance(lat1, lon1, lat2, lon2):
    R = 6373.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)

    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    #     distance = np.where(distance==0.0, -1, distance)
    thresh_distance = np.where(distance < 1, 1, distance)

    #     print("Result:", distance)
    return thresh_distance, distance


def clean_dataframe(train=True):
    if train:
        df = pd.read_csv("train.csv", index_col="tripid")
        inc_data = df[df['label'] == 'incorrect']
    else:
        df = pd.read_csv("test.csv", index_col="tripid")

    df['drop_time'] = pd.to_datetime(df['drop_time'])
    df['pickup_time'] = pd.to_datetime(df['pickup_time'])
    df['duration_minutes'] = (df['drop_time'] - df['pickup_time']) / pd.Timedelta(
        minutes=1)

    df['lat'] = abs(df['pick_lat'] - df['drop_lat'])
    df['lon'] = abs(df['pick_lon'] - df['drop_lon'])
    df['trip_distance'], df['acct_trip_distance'] = get_distance(df['pick_lat'], df['pick_lon'], df['drop_lat'],
                                                                 df['drop_lon'])

    df['mobile_fare'] = df['fare'] - df['additional_fare'] - df['meter_waiting_fare']
    df['mobile_time'] = df['duration'] - df['meter_waiting'] - df['meter_waiting_till_pickup']
    df['per_km'] = df['mobile_fare'] / df['trip_distance']

    df['fare'] = df['fare'].fillna(0)

    df['mock_time'] = df['mobile_time'].apply(lambda x: df['mobile_time'].mean() if x == 0 else x)
    df['speed'] = df['mobile_fare'] / df['mock_time']

    df['total_wait_time'] = df['meter_waiting'] + df['meter_waiting_till_pickup']

    # df['proportion_1'] = df['additional_fare'] / df['mobile_time'].apply(lambda x: 1 if x == 0 else x)
    # df['proportion_2'] = df['additional_fare'] / df['acct_trip_distance'].apply(lambda x: 1 if x == 0 else x)
    # df['proportion_3'] = df['duration_minutes'] / df['meter_waiting'].apply(lambda x: 1 if x == 0 else x)

    feature = df.drop(
        ['duration', 'pick_lat', 'pick_lon', 'drop_lat', 'drop_lon', 'pickup_time', 'drop_time',
         'trip_distance', 'mock_time'], axis=1)

    if train:
        label = (df['label'] == 'correct').astype('int')
        feature = feature.drop(columns=['label'])
        feature = robust_scaler.fit_transform(feature)
        return feature, label
    else:
        col_names = feature.columns
        feature = robust_scaler.transform(feature)
        return feature, col_names


def write_to_log(data, new_params=False):
    with open(f'logs/{run_num}_log.txt', 'a') as LOG_FILE:
        if new_params:
            LOG_FILE.write('*' * 50)

        for key, item in data.items():
            LOG_FILE.write(f'{key} : {item} \n')

        if len(data) == 1:
            LOG_FILE.write('_' * 50)
            LOG_FILE.write('\n')

        if new_params:
            LOG_FILE.write('*' * 50)


def record_history(data, new_activity=False):
    with open(f'logs/history.txt', 'a') as history:
        if new_activity:
            history.write('*' * 250 + '\n')
        else:
            history.write(str(data) + '\n')


def best_threshold(y_true, pred):
    max_f1 = 0
    thresh_val = 0
    for j in range(5, 6):
        threshold = j / 10
        thresh_pred = np.array((pred > threshold).astype('int'))
        f1_val = f1_score(y_true, thresh_pred)
        if f1_val > max_f1:
            thresh_val = threshold
            max_f1 = f1_val

    return max_f1, thresh_val


def f1_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    err = 1 - f1_score(y_true, np.round(y_pred))
    return 'f1_err', err


def tree_optimization(learning_rate, gamma, max_depth, subsample, reg_lambda, num_parallel_tree, min_child_weight):
    global tree_no, best_score

    X_train, X_eval, y_train, y_eval = train_test_split(features, labels, test_size=0.2, shuffle=True,
                                                        stratify=labels)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, shuffle=True,
                                                        stratify=y_train)
    booster_params = {
        'n_estimators': 500,
        'learning_rate': learning_rate,
        'gamma': gamma,
        'max_depth': int(np.around(max_depth)),
        'subsample': subsample,
        'sampling_method': 'gradient_based',
        'reg_lambda': reg_lambda,
        'min_child_weight': int(np.around(min_child_weight)),
        'num_parallel_tree': int(np.around(num_parallel_tree)),
        'objective': 'binary:logistic',
        'verbosity': 1,
        'max_delta_step ': 1
    }

    print("generating model")
    trip_model = XGBClassifier(**booster_params)
    trip_model.fit(X=X_train, y=y_train, eval_set=[(X_test, y_test)], eval_metric=f1_eval, early_stopping_rounds=25)

    print("Training Accuracy: %.2f" % (trip_model.score(X_eval, y_eval) * 100), "%")
    preds = trip_model.predict(X_eval)
    current_score = f1_score(y_eval, preds)

    if current_score > best_score:
        print(f"Score increased to {current_score} from {best_score}")
        best_score = current_score
        sub_pred = trip_model.predict(sub)
        trip_model.save_model(f'logs/f1_{run_num}_{best_score}_{tree_no}_thresh.model')
        savetxt(f'logs/{run_num}_{tree_no}_preds.txt', sub_pred, delimiter=',')

    booster_params['f1'] = current_score
    booster_params['tree_no'] = tree_no
    record_history(booster_params)
    tree_no += 1
    return current_score


def use_bayesian_optimization(update=False):
    max_file = open(f"logs/max_params_{run_num}.txt", 'w')
    if update:
        updated_space = update_values()
        search_space = {
            'learning_rate': updated_space['lr'],
            'gamma': updated_space['gamma'],
            'max_depth': updated_space['depth'],
            'min_child_weight': updated_space['child_weight'],
            'subsample': updated_space['sub_sample'],
            'reg_lambda': updated_space['lambda'],
            'num_parallel_tree': updated_space['num_trees'],
        }
    else:
        search_space = {
            'learning_rate': (0.01, 0.3),
            'gamma': (0, 1),
            'max_depth': (3, 12),
            'min_child_weight': (2, 7),
            'subsample': (0.6, 1),
            'reg_lambda': (0, 2),
            'num_parallel_tree': (7, 17),
        }
    NN_BAYESIAN = BayesianOptimization(tree_optimization, search_space)
    NN_BAYESIAN.maximize(init_points=50, n_iter=450, acq='ei', xi=0.0)
    write_to_log(NN_BAYESIAN.max)
    print('Best NN parameters: ', NN_BAYESIAN.max, file=max_file)


def create_tree():
    X_train, X_eval, y_train, y_eval = train_test_split(features, labels, test_size=0.2, shuffle=True,
                                                        stratify=labels, random_state=RANDOM_SEED)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, shuffle=True,
                                                        stratify=y_train, random_state=RANDOM_SEED)

    booster_params = {'n_estimators': 500, 'objective': 'binary:logistic', 'verbosity': 1}

    print("generating model")

    trip_model = XGBClassifier(**booster_params)
    trip_model.fit(X=X_train, y=y_train, eval_set=[(X_test, y_test)], eval_metric=f1_eval, early_stopping_rounds=50)

    iter_num = trip_model.best_iteration + 30
    booster_params = {'n_estimators': iter_num, 'objective': 'binary:logistic', 'verbosity': 1}
    trip_model = XGBClassifier(**booster_params)
    trip_model.fit(X=X_train, y=y_train, eval_set=[(X_test, y_test)], eval_metric=f1_eval)

    print("Training Accuracy: %.2f" % (trip_model.score(X_eval, y_eval) * 100), "%")
    preds = trip_model.predict(X_eval)
    current_score = f1_score(y_eval, preds)
    trip_model.save_model(f'logs/f1_{current_score}_thresh.model')
    sub_pred = trip_model.predict(sub)
    savetxt(f'logs/{current_score}_preds.txt', sub_pred, delimiter=',')

    return trip_model


def show_feature_importance(model_name):
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    colours = plt.cm.Set1(np.linspace(0, 1, 9))
    ax = plot_importance(model_name, height=1, color=colours, grid=False, show_values=False, importance_type='cover',
                         ax=ax)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)

    ax.set_xlabel('importance score', size=16)
    ax.set_ylabel('features', size=16)
    ax.set_yticklabels(ax.get_yticklabels(), size=12)
    ax.set_title('Ordering of features by importance to the model learnt', size=20)
    plt.show()


tree_no = 0
best_score = 0
run_num = 0

simple_imputer = SimpleImputer(strategy='most_frequent')
robust_scaler = RobustScaler()

features, labels = clean_dataframe()
sub, feature_names = clean_dataframe(train=False)


use_bayesian_optimization()
for run_num in range(1, 5):
    use_bayesian_optimization(update=True)
# model = create_tree()

# show_feature_importance(model)
