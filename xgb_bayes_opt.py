import numpy as np
from sklearn.metrics import f1_score
from numpy import savetxt
from bayes_opt import BayesianOptimization
from matplotlib import pyplot as plt
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
from update_search_space import update_values


def record_history(data, new_activity=False):
    with open(f'logs/history.txt', 'a') as history:
        if new_activity:
            history.write('*' * 250 + '\n')
        else:
            history.write(str(data) + '\n')


def show_feature_importance(model_name):
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    colours = plt.cm.Set1(np.linspace(0, 1, 9))
    ax = plot_importance(model_name, height=1, color=colours, grid=False, show_values=False,
                         importance_type='cover',
                         ax=ax)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)

    ax.set_xlabel('importance score', size=16)
    ax.set_ylabel('features', size=16)
    ax.set_yticklabels(ax.get_yticklabels(), size=12)
    ax.set_title('Ordering of features by importance to the model learnt', size=20)
    plt.show()


def f1_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    err = 1 - f1_score(y_true, np.round(y_pred))
    return 'f1_err', err


class XgbOptimizer(object):
    def __init__(self, features, labels, pred, seed, model, init_points=50, n_iter=450):
        self.tree_no = 0
        self.best_score = 0
        self.run_num = 0
        self.features = features
        self.labels = labels
        self.pred = pred
        self.seed = seed
        self.model_type = model
        self.init_points = init_points
        self.n_iter = n_iter

    def write_to_log(self, data, new_params=False):
        with open(f'logs/{self.run_num}_{self.model_type}_log.txt', 'a') as LOG_FILE:
            if new_params:
                LOG_FILE.write('*' * 50)

            for key, item in data.items():
                LOG_FILE.write(f'{key} : {item} \n')

            if len(data) == 1:
                LOG_FILE.write('_' * 50)
                LOG_FILE.write('\n')

            if new_params:
                LOG_FILE.write('*' * 50)

    def tree_optimization(self, learning_rate, gamma, max_depth, subsample, reg_lambda, num_parallel_tree,
                          min_child_weight):

        x_train, x_eval, y_train, y_eval = train_test_split(self.features, self.labels, test_size=0.2, shuffle=True,
                                                            stratify=self.labels, random_state=self.seed)

        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, shuffle=True,
                                                            stratify=y_train, random_state=self.seed)
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
        model = XGBClassifier(**booster_params)
        model.fit(X=x_train, y=y_train, eval_set=[(x_test, y_test)], eval_metric=f1_eval, early_stopping_rounds=25)

        preds = model.predict(x_eval)
        current_score = f1_score(y_eval, preds)

        print(f"eval ROC score: {current_score}")

        if current_score > self.best_score:
            print(f"Score increased to {current_score} from {self.best_score}")
            self.best_score = current_score
            sub_pred = model.predict(self.pred)
            model.save_model(
                f'logs/f1_{self.run_num}_{self.best_score}_{self.tree_no}_{self.model_type}_thresh.model')
            savetxt(f'logs/{self.run_num}_{self.tree_no}_{self.model_type}_preds.txt', sub_pred, delimiter=',')

        booster_params['f1'] = current_score
        booster_params['tree_no'] = self.tree_no
        record_history(booster_params)
        self.tree_no += 1
        return current_score

    def use_bayesian_optimization(self, update=False):
        max_file = open(f"logs/max_params_{self.run_num}_{self.model_type}.txt", 'w')
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
        NN_BAYESIAN = BayesianOptimization(self.tree_optimization, search_space)
        NN_BAYESIAN.maximize(init_points=self.init_points, n_iter=self.n_iter, acq='ei', xi=0.0)
        self.write_to_log(NN_BAYESIAN.max)
        print('Best NN parameters: ', NN_BAYESIAN.max, file=max_file)

    def create_tree(self):
        X_train, X_eval, y_train, y_eval = train_test_split(self.features, self.labels, test_size=0.2, shuffle=True,
                                                            stratify=self.labels, random_state=self.seed)

        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, shuffle=True,
                                                            stratify=y_train, random_state=self.seed)

        booster_params = {'n_estimators': 500, 'objective': 'binary:logistic', 'verbosity': 1}

        print("generating model")

        model = XGBClassifier(**booster_params)
        model.fit(X=X_train, y=y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=50)

        iter_num = model.best_iteration + 30
        booster_params = {'n_estimators': iter_num, 'objective': 'binary:logistic', 'verbosity': 1}
        model = XGBClassifier(**booster_params)
        model.fit(X=X_train, y=y_train, eval_set=[(X_test, y_test)])

        print("Training Accuracy: %.2f" % (model.score(X_eval, y_eval) * 100), "%")
        preds = model.predict(X_eval)
        current_score = f1_eval(y_eval, preds)
        model.save_model(f'logs/f1_{current_score}_thresh.model')
        sub_pred = model.predict(self.pred)
        savetxt(f'logs/{current_score}_preds.txt', sub_pred, delimiter=',')

        return model
