import io
import gc
import requests
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import time
from typing import Union, Dict
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def create_target(forex_pair:str,
                  trade_mode:str,
                  target_look_ahead:int,
                  target_take_profit:int,
                  target_stop_loss:int,
                  api_key: str):
  headers = {
    "Authorization":api_key

  }
  data = {
      'trade_mode': trade_mode,
      'forex_pair': forex_pair,
      "look_ahead": target_look_ahead,
      "take_profit": target_take_profit,
      "stop_loss": target_stop_loss,
  }

  res = requests.post("http://127.0.0.1:8001/target/forex",headers=headers,json=data,)
  target_token = res.json()["target token"]
  headers = {
    "Authorization":api_key
  }
  data = {
    'target_token': target_token,
  }

  res = requests.get("http://127.0.0.1:8001/get_target",headers=headers,params=data,)
  target_name = f"trg_clf_{trade_mode}_{forex_pair}_M{target_look_ahead}_TP{target_take_profit}_SL{target_stop_loss}"
  open(f"./{target_name}.parquet", 'wb').write(res.content)
  
  return target_token, target_name

def get_features(forex_pair:str, api_key: str):

    headers = {
    "Authorization":api_key
    }
    params = {
        'symbol_pair': forex_pair,
    }

    res = requests.get("http://127.0.0.1:8001/get_dataset",headers=headers,params=params,)
    open(f"./{forex_pair}.parquet", 'wb').write(res.content)

def create_TS_cross_val_folds(
    df_all :pd.DataFrame,
    max_train_size : int,
    n_splits: int,
    test_size: int,
    train_test_gap_size : int = 0,
    early_stopping_rounds : Union[int,None]  = None,
):
    """
    Return a nested dictionary key is k number and value is dicitonary of train, valid and test Dates
    :max_train_size: maximum size we for train
    :n_splits: K in cross-folds
    :test_size: test size
    """
    all_dates = df_all.index.get_level_values("_time").unique().sort_values("_time")
    tscv = TimeSeriesSplit(
        gap = train_test_gap_size,
        max_train_size=max_train_size + test_size,
        n_splits=n_splits,
        test_size=test_size,
    )
    folds = {}

    if early_stopping_rounds is not None:
        for i, (train_valid_index, test_index) in enumerate(tscv.split(all_dates[0])):
            folds[i] = {
                "train_dates": all_dates[0][train_valid_index[:-test_size]],
                "valid_dates": all_dates[0][train_valid_index[-test_size:]],
                "test_dates": all_dates[0][test_index],
            }
    else:

        for i, (train_valid_index, test_index) in enumerate(tscv.split(all_dates[0])):
            folds[i] = {
                "train_dates": all_dates[0][train_valid_index],
                "test_dates": all_dates[0][test_index],
            }



    return folds

def cal_eval(y_real, y_pred):
    """
    calculate and return evaluation of class 1

    """
    
    tn, fp, fn, tp = confusion_matrix(y_real, y_pred, labels=[0, 1]).ravel()
    clf_report = classification_report(
        y_real, y_pred, output_dict=True, digits=0, zero_division=0,target_names=[0,1]
    )
    if 1 not in clf_report.keys():
        print("NO CLASS1 PREDICTION !")
        clf_report[1] = {"f1-score": 0.0, "precision": 0.0, "recall": 0.0}

    class_1_report = clf_report[1]
    for k, v in class_1_report.items():
        class_1_report[k] = round(v, 2)

    eval_list = [
        class_1_report["f1-score"],
        class_1_report["precision"],
        class_1_report["recall"],
        tp,
        fp,
        tn,
        fn,
    ]
    return eval_list

def run_model_on_folds(df, folds, model, early_stopping_rounds):
    evals = pd.DataFrame(
        columns=[
            "dataset",
            "K",
            "f1_score",
            "precision",
            "recall",
            "TP",
            "FP",
            "TN",
            "FN",
            "Min_date",
            "Max_date",
            "train_duration",
        ]
    )
    df["model_prediction"] = -1
    df["model_prediction_proba"] = -1
    df["K"] = -1
    non_feture_cols = [f for f in list(df.columns) if 'feature_' not in f]

    input_cols = [f for f in list(df.columns) if f not in non_feture_cols]
    print(f"number of features: {len(input_cols)}")

    for i in list(folds.keys()):
        print("="*40)
        print(f"Fold {i}:")
        tic = time.time()

        min_max_dates = {}
        for d in list(folds[i].keys()):
            min_max_dates.update({
                d:[folds[i][d].min(), folds[i][d].max()]
            })
            print(f"--> fold {d} size: {df.loc[folds[i][d]].shape}")
            

        if early_stopping_rounds is not None:
            print("early_stopping_rounds: ", early_stopping_rounds)
            eval_set = [
                (
                    df.loc[folds[i]["valid_dates"]].drop(
                        columns=non_feture_cols
                    ),
                    df.loc[folds[i]["valid_dates"]]["target"],
                )
            ]

            model.fit(
                df.loc[folds[i]["train_dates"]].drop(
                    columns=non_feture_cols
                ),
                df.loc[folds[i]["train_dates"]]["target"],
                eval_set=eval_set,
                verbose=False,
            )

        else:

            model.fit(
                df.loc[folds[i]["train_dates"]].drop(
                    columns=non_feture_cols
                ),
                df.loc[folds[i]["train_dates"]]["target"],
            )


        toc = time.time()
        gc.collect()
        # repetetive part I can improve by a function
        for set_name in list(folds[i].keys()):

            y_pred = model.predict(df.loc[folds[i][set_name]][input_cols]).reshape(
                -1, 1
            )

            y_real = df.loc[folds[i][set_name]][["target"]]

            eval_list = (
                [set_name.replace("_dates",""), i]
                + cal_eval(y_real=y_real, y_pred=y_pred)
                + min_max_dates[set_name]
                + [str(round(toc - tic, 1))]
            )

            evals.loc[len(evals)] = eval_list

            if set_name == "test_dates":
                df.loc[folds[i]["test_dates"], "K"] = i
                df.loc[folds[i]["test_dates"], "model_prediction"] = y_pred
                proba_pred = model.predict_proba(df.loc[folds[i][set_name]][input_cols])

                if np.shape(proba_pred)[1] > 1:
                    df.loc[
                        folds[i]["test_dates"], "model_prediction_proba"
                    ] = model.predict_proba(df.loc[folds[i][set_name]][input_cols])[
                        :, 1
                    ]
                else:
                    print("Proba doesn't have class1")
                    df.loc[folds[i]["test_dates"], "model_prediction_proba"] = 0

        print(evals.iloc[-len(folds[i].keys()):])

    return (
        evals,
        df[df.model_prediction != -1][["K", "model_prediction", "model_prediction_proba", "target"]],
    )

def backtest_strategy(df_model_signal: pd.DataFrame, strategy_config:Dict[str,Union[int,str]], api_key:str):
    buffer = io.BytesIO()
    df_model_signal.to_parquet(buffer)
    parquet_bytes = buffer.getvalue()

    headers = {"Authorization": api_key}

    # Create files dictionary with in-memory Parquet bytes
    files = {"df_model_signal_file": ("df_model_signal.parquet", parquet_bytes)}

    res = requests.post("http://127.0.0.1:8001/backtest", headers=headers, params=strategy_config, files=files)
    return res