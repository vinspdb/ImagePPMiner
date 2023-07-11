from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.importer.xes import importer as xes_importer
import numpy as np

class ImagePPMiner:
    def __init__(self, eventlog):
        self._eventlog = eventlog

    def import_log(self):
        log = xes_importer.apply('dataset/'+self._eventlog+'.xes')
        dataframe = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)

        unique = dataframe['concept:name'].unique()
        dictOfWords = { i : unique[i] for i in range(0, len(unique) ) }
        dictOfWords = {v: k for k, v in dictOfWords.items()}
        for k in dictOfWords:
            dictOfWords[k] += 1
        dataframe['concept:name'] = [dictOfWords[item] for item in dataframe['concept:name']]
        dataframe = dataframe[["case:concept:name", "concept:name", "time:timestamp"]]
        return dataframe

    def generate_prefix_trace(self, log, n_caseid):
        grouped = log.groupby("case:concept:name")
        start_timestamps = grouped["time:timestamp"].min().reset_index()
        start_timestamps = start_timestamps.sort_values("time:timestamp", ascending=True, kind="mergesort")
        train_ids = list(start_timestamps["case:concept:name"])[:int(0.66 * len(start_timestamps))]
        train = log[log["case:concept:name"].isin(train_ids)].sort_values("time:timestamp", ascending=True,kind='mergesort')
        test = log[~log["case:concept:name"].isin(train_ids)].sort_values("time:timestamp", ascending=True,kind='mergesort')
        
        train_act = train.groupby('case:concept:name', sort=False).agg({'concept:name': lambda x: list(x)})
        train_temp = train.groupby('case:concept:name', sort=False).agg({'time:timestamp': lambda x: list(x)})
        test_act = test.groupby('case:concept:name', sort=False).agg({'concept:name': lambda x: list(x)})
        test_temp = test.groupby('case:concept:name', sort=False).agg({'time:timestamp': lambda x: list(x)})

        return train_act, train_temp, test_act, test_temp

    @staticmethod
    def generate_image(act_val, time_val, max_trace, n_activity):
        i = 0
        matrix_zero = [max_trace, n_activity, 2]
        image = np.zeros(matrix_zero)
        list_image = []
        while i < len(time_val):
            j = 0
            list_act = []
            list_temp = []
            a = list(range(1, n_activity + 1))
            dict_act = dict.fromkeys(a, 0)
            dict_time = dict.fromkeys(a, 0)
            while j < (len(act_val.iat[i, 0]) - 1):
                start_trace = time_val.iat[i, 0][0]
                dict_act[act_val.iat[i, 0][0 + j]] += 1
                duration = time_val.iat[i, 0][0 + j] - start_trace
                days = (duration.total_seconds())/86400
                dict_time[act_val.iat[i, 0][0 + j]] = days
                l_act = list(dict_act.values())
                l_time = list(dict_time.values())
                list_act.append(l_act)
                list_temp.append(l_time)
                j = j + 1
            cont = 0
            lenk = len(list_act)
            image = np.zeros(matrix_zero)
            while cont < lenk:
                z = 0
                image = np.delete(image,0,0)
                image = np.append(image, np.zeros([1, n_activity, 2]), axis=0)
                while z < n_activity:
                    image[(max_trace - 1)][z] = [list_act[cont][z], list_temp[cont][z]]
                    z = z + 1
                cont = cont + 1
                list_image.append(image)
            i = i + 1
        return list_image


    @staticmethod
    def get_label(act):
        i = 0
        list_label = []
        while i < len(act):
            j = 0
            while j < (len(act.iat[i, 0]) - 1):
                list_label.append(act.iat[i, 0][j + 1])
                j = j + 1
            i = i + 1
        return list_label

    @staticmethod
    def dataset_summary(log):
        print("Activity Distribution\n", log['concept:name'].value_counts())
        n_caseid = log['case:concept:name'].nunique()
        n_activity = log['concept:name'].nunique()
        print("Number of CaseID", n_caseid)
        print("Number of Unique Activities", n_activity)
        print("Number of Activities", log['concept:name'].count())
        cont_trace = log['case:concept:name'].value_counts(dropna=False)
        max_trace = max(cont_trace)
        print("Max lenght trace", max_trace)
        print("Mean lenght trace", np.mean(cont_trace))
        print("Min lenght trace", min(cont_trace))
        return max_trace, n_caseid, n_activity
