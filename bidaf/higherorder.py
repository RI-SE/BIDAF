import re
import pandas as pd
from collections import defaultdict
from scipy.spatial.distance import cosine

def group_by_scale(labels):
    """ Utility that groups attribute labels by time scale """
    groups = defaultdict(list)
    # Extract scales from labels (assumes that the scale is given by the last numeral in a label)
    for s in labels:
        m = re.findall("\d+", s)
        if m:
            groups[m[-1]].append(s)
        else:
            print("Bad attribute: ", s)
    return list(groups.values())

class HigherOrderModel():
    """ Higher-order representations module """
    def __init__(self):
        self.version = 0
        self.update_first_interval = 600
        self.update_interval = 200
        self.data = []
        self.type = "HigherOrderModel"
        self.relations = []
        self.scale_groups = []
        self.reserved_attributes = []
        self.has_time = False

    def model_version(self):
        return self.version

    def model_type(self):
        return self.type

    def default_params(self):
        return {"update_interval": 100,
                "index_attribute": False,
                "time_attribute": False,
                "entity_attribute": False}

    def set_params(self, params):
        if "update_interval" in params:
            self.update_interval = params["update_interval"]
        if "index_attribute" in params and params["index_attribute"] is not False:
            self.reserved_attributes.append(params["index_attribute"])
        if "time_attribute" in params and params["time_attribute"] is not False:
            self.reserved_attributes.append(params["time_attribute"])
            self.has_time = True
        if "entity_attribute" in params and params["entity_attribute"] is not False:
            self.reserved_attributes.append(params["entity_attribute"])

    def extract_model(self):
        return {"type": self.type,
                "version": self.version,
                "relations": self.relations}

    def update_relations(self):
        # Convert dictionary list to dataframe
        data_df = pd.DataFrame(self.data).drop(self.reserved_attributes, axis=1)
        # Center and normalize
        data_df = (data_df-data_df.mean())/data_df.std()
        # Group attributes by scale
        if self.scale_groups == []:
            if self.has_time:
                self.scale_groups = group_by_scale(list(data_df))
            else:
                self.scale_groups = [list(data_df)]
        # Calculate similarities per scale
        similarities = []
        for group in self.scale_groups:
            group_df = data_df[group]
            # Calculate pairwise correlation coefficients
            correlations = group_df.corr()
            # Calculate pairwise similarities
            group_size = len(group)
            for i in range(group_size-1):
                for j in range(i+1, group_size):
                    a1, a2 = group[i], group[j]
                    # Get correlation vectors of a1 and a2 and drop missing values
                    c = correlations[[a1, a2]].dropna()
                    if a1 in c[a2]:
                        # Cosine similarity between vectors discounted by correlation
                        s = 1 - cosine(c[a1], c[a2]) - c[a1][a2]
                        similarities.append(((a1, a2) if a1 < a2 else (a2, a1), s))
        # Sort by similarity in descending order
        self.relations = sorted(similarities, key=lambda v: -v[1])

    def handle_data(self, sample):
        # Accumulate data
        self.data.append(sample.copy())
        # Update model at regular intervals
        if len(self.data) % self.update_interval == 0 and len(self.data) >= self.update_first_interval:
            # Re-calculate higher-order relations
            self.update_relations()
            # Update version number
            self.version += 1

    def handle_batch_data(self, data):
        # Accumulate data
        self.data += data
        # Calculate higher-order relations
        self.update_relations()
        # Update version number
        self.version += 1

    def features_changed(self):
        return False

    def print_relations(self):
        t = [(s, a, b) for ((a, b), s) in self.relations]
        c = ["Similarity", "First attribute", "Second attribute"]
        print(pd.DataFrame(t, columns=c).to_string(index=False))
