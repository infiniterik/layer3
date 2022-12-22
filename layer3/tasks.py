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
    noise = {"\u00a0", "&gt;"}

    def remove_noise(self, text):
        return " ".join([x for x in text.split("\n") if x and x not in self.noise])

    def get_post(self, element):
        text = ""
        if type(element.title) is str:
            text = element.title+"\n"
        if type(element.selftext) is str:
            text += element.selftext
        if type(element.body) is str:
            text += element.body
        return self.remove_noise(text)
        
    def source_text(self, element, data):
        return f'Classify Post: {self.post(element)}'
    
    def target_text(self, element, data):
        return element.subreddit

class EnrichmentTask(ClassifyTask):
    def __init__(self, enrichment_id=None, levels=None, pre_filter_strategy=PostsWithTitles(), post_filter_strategy=RemoveEmptyPost()):
        super().__init__(pre_filter_strategy, post_filter_strategy)
        self.enrichment_id = enrichment_id
        if not levels:
            self.levels = True
            self.enrichment = [("low", 0.5), ("high", 1.0)]
        else:
            self.levels = True
            self.enrichment = sorted([x for x in levels.items()], key=lambda x: x[1])
    
    def pre_hook(self, data):
        if self.levels:
            return
        df = data["enrichments"].apply(lambda x: x[self.enrichment_id]).sort_values()
        low = df.iloc[len(df)//3]
        medium = df.iloc[2*(len(df)//3)]
        self.enrichment = [("low", low), ("medium", medium), ("high", 1.0)]
    
    def post_hook(self, data):
        if self.levels:
            return
        self.toxicity = None

    def enrichment_level(self, element):
        for i in range(len(self.enrichment)):
            if self.enrichment[i][1] >= element.enrichments[self.enrichment_id]:
                return self.enrichment[i][0]
        return self.enrichment[-1][0]

    def source_text(self, element, data):
        return f'{self.enrichment_id}: {self.post(element)}'

    def target_text(self, element, data):
        return f"{self.enrichment_level(element)}"

class ParentPostTask(ClassifyTask):
    name = "ParentPostTask"
    def __init__(self, pre_filter_strategy=FilterData()):
        super().__init__(pre_filter_strategy=pre_filter_strategy)
        
    def pre_hook(self, data):
        self.parents = {}
        for d in data.itertuples(index=False):
            self.parents[d.id] = d
    
    def post_hook(self, data):
        self.parents = {}
    
    def get_parent(self, element):
        return self.parents.get(element.parent_id.split("_")[-1], None)
    
    def source_text(self, element, data):
        return f'Parent {element.subreddit}: {self.post(element)}'
    
    def target_text(self, element, data):
        if not element.parent_id:
            return ""
        parent = self.get_parent(element)
        if parent is not None:
            return f"{self.post(parent)}"
        return ""

class DesiredEnrichmentTask(ParentPostTask, EnrichmentTask):
    name = "DesiredEnrichmentTask"
    def __init__(self, enrichment_id, pre_filter_strategy=And(RemoveRemoved(), PostsWithTitles())):
        super().__init__(pre_filter_strategy=pre_filter_strategy)
        self.enrichment_id = enrichment_id

    def source_text(self, element, data):
        parent = None
        if element.parent_id:
            parent = self.get_parent(element)
            if parent:
                t = self.post(parent)
            else:
                return ""
        elif element.title:
            t = element.title
        else:
            return ""
        return f"{self.enrichment_id}: {self.enrichment_level(element)} Parent: {t}"
    
    def target_text(self, element, data):
        return f"{self.post(element)}"