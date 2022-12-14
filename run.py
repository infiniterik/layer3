from layer3 import data
from layer3.tasks import DesiredEnrichmentTask
from layer3.filters import RemoveRemoved

ctask = DesiredEnrichmentTask(enrichment_id="anger")#pre_filter_strategy=RemoveRemoved()) #ClassifyTask()

d = data.process(["test.json"], ctask, train_size=5, test_size=5)
data.write(d, "test_out.json")