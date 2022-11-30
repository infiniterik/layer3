import pandas as pd
import json

def load_data(fname, lines=True):
    # TODO: more complex loading options
    data = pd.read_json(fname, lines=lines)
    return data

def get_parent(element, data):
    if element.parent_id:
        id = element.parent_id.split("_")[-1]
        result = data[(data.id == id)]
        if not result.empty:
            return result.iloc[0]
    return None

def process(files, task_object, train_size, test_size):
    # Add 2000 instances for testing
    k = (train_size+test_size)//(len(files))
    print(f"taking {k} lines from each file")

    #ctask = ParentPostTask(pre_filter_strategy=RemoveRemoved()) #ClassifyTask()
    loaded = {f : load_data(f) for f in files}
    data = [task_object(d, batch_name=f).sample(n=k) for f, d in loaded.items()]

    data = pd.concat(data).sample(frac=1, random_state=1).reset_index()
    train= data.iloc[:train_size,:]
    test = data.iloc[train_size:,:]
    return dict(train=train, test=test)

def write(dataset, fname):
    dataset = {f:d.to_dict(orient="records") for f, d in dataset.items()}
    with open(fname, 'w') as out:
        json.dump(dataset, out)