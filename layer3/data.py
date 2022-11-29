import pandas as pd
from tqdm.auto import tqdm
from collections import defaultdict as ddict
import json

tqdm.pandas()

def load_data(fname, lines=True):
    """Want to filter out [removed] posts, duplicate posts"""
    data = pd.read_json(fname, lines=lines)
    return data

def get_parent(element, data):
    if element.parent_id:
        id = element.parent_id.split("_")[-1]
        result = data[(data.id == id)]
        if not result.empty:
            return result.iloc[0]
    return None

class FilterData:
    name = "FilterData"
    def filter(self, data):
        return data
    
    def __call__(self, data):
        return self.filter(data)
    
class PostsWithTitles(FilterData):
    name = "PostsWithTitles"
    def filter(self, data):
        return data[(data.title != "")].loc[(data.selftext != "") | (data.body != "")].loc[(data.selftext != "[removed]") & (data.body != "[removed]")]

class RemoveRemoved(FilterData):
    name = "RemoveRemoved"
    def filter(self, data):
        return data[(data.body != "[removed]") & (data.selftext != "[removed]")]

class RemoveEmptyPost(FilterData):
    name = "RemoveEmpty"
    def filter(self, data):
        return data[(data.target_text != "") & (data.source_text != "")]
    
class PrepareData:
    name = "PrepareData"
    def __init__(self, pre_filter_strategy=PostsWithTitles(), post_filter_strategy=RemoveEmptyPost()):
        self.pre_filter_strategy=pre_filter_strategy
        self.post_filter_strategy=post_filter_strategy
        self.state = None
    
    def source_text(self, element, data):
        return element
    
    def target_text(self, element, data):
        return element
        
    def pre_filter(self, data, batch_name):
        d = len(data)
        data = self.pre_filter_strategy(data)
        print(f"{self.pre_filter_strategy.name}|{batch_name}: len: {d} -> len: {len(data)}")
        return data
    
    def post_filter(self, data, batch_name):
        d = len(data)
        data = self.post_filter_strategy(data)
        print(f"{self.post_filter_strategy.name}|{batch_name}: len: {d} -> len: {len(data)}")
        return data
    
    def pre_hook(self, data):
        pass
    
    def post_hook(self, data):
        self.state = None
    
    def __call__(self, data, batch_name="PrepareData"):
        df = pd.DataFrame()
        self.pre_hook(data)
        data = self.pre_filter(data, batch_name)
        
        tqdm.pandas(desc=f"{self.name}:{batch_name} - source_text")
        df["source_text"] = data.progress_apply(lambda x: self.source_text(x, data), axis=1)
        
        tqdm.pandas(desc=f"{self.name}:{batch_name} - target_text")
        df["target_text"] = data.progress_apply(lambda x: self.target_text(x, data), axis=1)
        
        df = self.post_filter(df, batch_name)
        
        self.post_hook(data)
        return df#.drop(columns=self.drop)

class ClassifyTask(PrepareData):
    name = "ClassifyTask"
    #def __init__(self, filter_strategy=PostsWithTitles()):
    #    super().__init__(filter_strategy)
        
    def source_text(self, element, data):
        return f'Classify Post: {element.title} {element.selftext} {element.body}'
    
    def target_text(self, element, data):
        return element.selftext + element.body

class ParentPostTask(PrepareData):
    name = "ParentPostTask"
    def __init__(self, pre_filter_strategy=FilterData()):
        super().__init__(pre_filter_strategy=pre_filter_strategy)
        
    def pre_hook(self, data):
        self.state = {}
        for d in data.itertuples(index=False):
            self.state[d.id] = d
    
    def source_text(self, element, data):
        return f'Parent {element.subreddit}: {element.selftext}{element.body}'
    
    def target_text(self, element, data):
        if element.parent_id == "":
            return ""
        parent = self.state.get(element.parent_id.split("_")[-1], None)
        if parent is not None:
            return f"{parent.title} {parent.selftext} {parent.body}"
        return ""


class GroupBy:
    name = "GroupBy"
    def group(self, data):
        return data
    
    def __call__(self, data):
        return self.group(data)

class GroupByPost(GroupBy):
    """
    Step 1: Sort by date created
    Step 2: link each post to root parent
    """
    def group(self, data):
        data = data.sort_values(by=["created"])
        parent_map = dict()
        thread = ddict(list)
        
        for row in tqdm(data.itertuples(index=False)):
            if row.parent_id:
                parent = row.parent_id.split("_")[1]
                if parent in parent_map:
                    parent = parent_map[parent]
                else:
                    parent_map[row.id] = parent
                thread[parent].append(row._replace(parent_id=parent))
            else: # Root post
                thread[row.id].append(row._replace(parent_id=row.id))
        #print(thread.keys())
        return [pd.DataFrame(t) for t in thread.values()]

def process(files, task_object, train_size, test_size):
    # Add 2000 instances for testing
    k = (train_size+test_size)//(len(files))

    ctask = ParentPostTask(pre_filter_strategy=RemoveRemoved()) #ClassifyTask()
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