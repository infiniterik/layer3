from layer3 import data
from layer3.tasks import ParentPostTask
from layer3.filters import RemoveRemoved

ctask = ParentPostTask(pre_filter_strategy=RemoveRemoved()) #ClassifyTask()

d = data.process(["test.json"], ctask, train_size=100, test_size=100)
data.write(d, "test_out.json")