import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from .filters import *
from .groupby import *

tqdm.pandas()

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

    def post(self, element):
        text = element.title
        if not text:
            text = ""
        text = element.selftext
        if type(element.selftext) is not str:
            text = element.body
        if type(text) is not str:
            text = ""
        return text
        
    def source_text(self, element, data):
        return f'Classify Post: {self.post(element)}'
    
    def target_text(self, element, data):
        return element.subreddit

class ToxicityTask(ClassifyTask):
    def __init__(self, levels=None, pre_filter_strategy=PostsWithTitles(), post_filter_strategy=RemoveEmptyPost()):
        super().__init__(pre_filter_strategy, post_filter_strategy)
        if not levels:
            self.levels = False
        else:
            self.levels = True
            self.state = sorted([x for x in levels.items()], key=lambda x: x[1])
    
    def pre_hook(self, data):
        if self.levels:
            return
        df = data["enrichments"].apply(lambda x: x["toxic"]).sort_values()
        low = df.iloc[len(df)//3]
        medium = df.iloc[2*(len(df)//3)]
        self.state = [("low", low), ("medium", medium), ("high", 1.0)]
    
    def post_hook(self, data):
        if self.levels:
            return
        self.state = None

    def tox_level(self, element):
        for i in range(len(self.levels)):
            if self.state[i][1] < element.enrichments["toxic"]:
                return self.state[i-1][0]
        return self.state[-1][0]

    def source_text(self, element, data):
        return f'Toxicity: {self.post(element)}'

    def target_text(self, element, data):
        return f"{self.tox_level(element)}"

class ParentPostTask(PrepareData):
    name = "ParentPostTask"
    def __init__(self, pre_filter_strategy=FilterData()):
        super().__init__(pre_filter_strategy=pre_filter_strategy)
        
    def pre_hook(self, data):
        self.state = {}
        for d in data.itertuples(index=False):
            self.state[d.id] = d
    
    def source_text(self, element, data):
        return f'Parent {element.subreddit}: {element.selftext}{element.body} </s>'
    
    def target_text(self, element, data):
        if element.parent_id == "":
            return ""
        parent = self.state.get(element.parent_id.split("_")[-1], None)
        if parent is not None:
            return f"{parent.title} {parent.selftext} {parent.body} </s>"
        return ""