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
 
class And(FilterData):
    def __init__(self, f1, f2):
        self.f1 = f1
        self.f2 = f2
        self.name = f"{self.f2.name} AND {self.f1.name}"
    
    def filter(self, data):
        return self.f1.filter(self.f2.filter(data))
