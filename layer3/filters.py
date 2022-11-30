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
 