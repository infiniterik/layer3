from collections import defaultdict as ddict
from tqdm import tqdm


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