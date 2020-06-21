import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from numpy import savetxt
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split
from xgboost import plot_importance
from matplotlib import pyplot as plt


def change_date(date_time_str):
    date_time_obj = datetime.datetime.strptime(date_time_str, '%m/%d/%Y %H:%M')
    d_time = date_time_obj.time()
    return [(d_time.hour * 60 + d_time.minute), date_time_obj.weekday()]


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


def prepare_dataframe(df):
    global imp_freq
    # pickup_time = np.array([change_date(x) for x in df['pickup_time'].values])
    # df['pickup_time'], df['week_day'] = pickup_time[:, 0], pickup_time[:, 1]

    # drop_time = np.array([change_date(x) for x in df['drop_time'].values])
    # df['drop_time'] = drop_time[:, 0]

    # df['lat_dif'] = np.abs(df['pick_lat'] - df['drop_lat'])
    # df['lon_dif'] = np.abs(df['pick_lon'] - df['drop_lon'])

    df['trip_time'] = df['duration'] - df['meter_waiting_till_pickup'] - df['meter_waiting']
    df['mobile_fare'] = df['fare'] - df['additional_fare'] - df['meter_waiting_fare']

    df['trip_distance'], df['acct_trip_distance'] = get_distance(df['pick_lat'], df['pick_lon'], df['drop_lat'],
                                                                 df['drop_lon'])
    df['per_km'] = df['mobile_fare'] / df['trip_distance']
    df['duration'] = df['duration'] / 60

    df = pd.DataFrame(imp_freq.fit_transform(df), columns=df.columns)

    # df.reset_index(drop=True, inplace=True)
    # df = pd.concat([df, pd.get_dummies(df['week_day'].values)], axis=1)
    #
    # bins = [x * 60 for x in range(24)]
    # df['time_bin'] = np.searchsorted(bins, df['pickup_time'].values)

    return df


def scale_data(df, train=True):
    global robust_scaler

    if train:
        x_scaled = robust_scaler.fit_transform(df.values)
        scaled = pd.DataFrame(x_scaled, columns=df.columns)

    else:
        x_scaled = robust_scaler.transform(df.values)
        scaled = pd.DataFrame(x_scaled, columns=df.columns)

    return scaled


def write_to_log(data, new_params=False):
    with open(f'logs/log.txt', 'a') as LOG_FILE:
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


def get_cleaned_df(train=True):
    if train:
        df = pd.read_csv("train.csv")
        df.dropna(inplace=True)

        df = prepare_dataframe(df)

        labels = df['label'] == 'correct'
        labels = labels.astype('int')
        df = df.drop(
            columns=['pick_lat', 'pick_lon', 'drop_lat', 'drop_lon', 'trip_distance', 'tripid', 'label', 'pickup_time',
                     'drop_time'])

        scaled_df = scale_data(df)

        return scaled_df, labels

    else:
        df = pd.read_csv("test.csv")
        df = prepare_dataframe(df)
        df = df.drop(
            columns=['pick_lat', 'pick_lon', 'drop_lat', 'drop_lon', 'trip_distance', 'tripid', 'pickup_time',
                     'drop_time'])
        scaled_df = scale_data(df, train=False)

        return scaled_df


def tree_optimization(eta, gamma, max_depth, subsample, lambda_val, num_parallel_tree, min_child_weight,
                      colsample_bytree):
    global trip_train, trip_test, tree_no, best_score

    booster_params = {
        'eta': eta,
        'gamma': gamma,
        "eval_metric": "error",
        'max_depth': int(np.around(max_depth)),
        'subsample': subsample,
        'sampling_method': 'gradient_based',
        'lambda': lambda_val,
        'min_child_weight': int(np.around(min_child_weight)),
        'num_parallel_tree': int(np.around(num_parallel_tree)),
        'objective': 'binary:logistic',
        'verbosity': 1,
        'max_delta_step ': 1,
        'colsample_bytree': colsample_bytree
    }

    results = {}

    print("generating model")

    trip_model = xgb.cv(booster_params, dtrain=trip_train, num_boost_round=1000, nfold=8, stratified=True,
                        early_stopping_rounds=10, verbose_eval=10, show_stdv=True)

    num_iter = trip_model.shape[0]
    current_score = - trip_model.iloc[-1]['test-error-mean']

    if current_score > best_score:
        print(f"Score increased to {current_score} from {best_score}")
        best_score = current_score
        trip_model = xgb.train(booster_params, trip_train, num_boost_round=num_iter, evals=[(trip_test, 'val')],
                               evals_result=results)
        train_pred = trip_model.predict(trip_test)
        max_f1, thresh_val = best_threshold(y_eval, train_pred)

        pred = trip_model.predict(trip_sub)

        trip_model.save_model(f'logs/f1_{max_f1}_{tree_no}_thresh_{thresh_val}.model')
        savetxt(f'logs/{tree_no}_preds.txt', pred, delimiter=',')

    booster_params['loss'] = current_score
    booster_params['tree_no'] = tree_no
    record_history(booster_params)
    tree_no += 1
    return current_score


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


def use_bayesian_optimization():
    max_file = open(f"logs/max_params.txt", 'w')

    NN_BAYESIAN = BayesianOptimization(
        tree_optimization,
        {
            'eta': (0.01, 0.3),
            'gamma': (0, 0.5),
            'max_depth': (3, 12),
            'min_child_weight': (2, 7),
            'subsample': (0.6, 1),
            'lambda_val': (0, 2),
            'num_parallel_tree': (7, 17),
            'colsample_bytree': (0.6, 1)
        })
    NN_BAYESIAN.maximize(init_points=20, n_iter=550, acq='ei', xi=0.0)
    write_to_log(NN_BAYESIAN.max)
    print('Best NN parameters: ', NN_BAYESIAN.max, file=max_file)


tree_no = 0
best_score = -1
robust_scaler = preprocessing.RobustScaler()
imp_freq = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
train_df, train_labels = get_cleaned_df()

print(train_df.columns)
sub_data = get_cleaned_df(False)

X_train, X_eval, y_train, y_eval = train_test_split(
    train_df.values,
    train_labels.values,
    test_size=0.2,
    shuffle=True,
    stratify=train_labels,
    random_state=1234
)

trip_train = xgb.DMatrix(X_train, label=y_train)
trip_test = xgb.DMatrix(X_eval, label=y_eval)
trip_sub = xgb.DMatrix(sub_data.values)

# use_bayesian_optimization()
#
# booster_params = {'eta': 0.33456455562233345, 'gamma': 0.12868071574700746, 'eval_metric': 'error', 'max_depth': 8,
#                   'subsample': 0.9576426911043358, 'sampling_method': 'gradient_based', 'lambda': 1.3952953897211688,
#                   'min_child_weight': 2, 'num_parallel_tree': 17, 'objective': 'binary:logistic', 'verbosity': 1,
#                   'max_delta_step ': 1, 'validate_parameters': 1}
booster_params = {'objective': 'binary:logistic', 'verbosity': 1}
#
#
trip_model = xgb.cv(booster_params, dtrain=trip_train, num_boost_round=1000, nfold=5, stratified=True,
                    early_stopping_rounds=10, verbose_eval=1, show_stdv=True)
num_iter = trip_model.shape[0]
current_score = - trip_model.iloc[-1]['test-error-mean']
trip_model = xgb.train(booster_params, trip_train, num_boost_round=num_iter, evals=[(trip_test, 'val')])
train_pred = trip_model.predict(trip_test)
sub_pred = trip_model.predict(sub_data)
thresh_pred = np.array((sub_pred > 0.5).astype('int'))
ride_pred = np.round(thresh_pred).astype('int')
max_f1, thresh_val = best_threshold(y_eval, train_pred)
print(max_f1, thresh_val)

fig = plt.figure(figsize=(14, 9))
ax = fig.add_subplot(111)
colours = plt.cm.Set1(np.linspace(0, 1, 9))
ax = plot_importance(trip_model, height=1, color=colours, grid=False, show_values=False, importance_type='cover',
                     ax=ax)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(2)

ax.set_xlabel('importance score', size=16)
ax.set_ylabel('features', size=16)
ax.set_yticklabels(ax.get_yticklabels(), size=12)
ax.set_title('Ordering of features by importance to the model learnt', size=20)
plt.show()
