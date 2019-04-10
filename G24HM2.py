from pyspark import SparkContext, SparkConf
import sys
import os

# 1. Data loading
# Import the Dataset
docs_txt = []

assert len(sys.argv) == 3, "Usage: python G24HM2.py <K> <file_name>"

K = sys.argv[1]
data_path = sys.argv[2]

assert K.isdigit(), "K must be an integer"
assert os.path.isfile(data_path) or os.path.isdir(data_path), "File or folder not found"

# Spark Setup
conf = SparkConf().setAppName('Word count 1&2').setMaster("local")
sc = SparkContext(conf)

# Create a parallel collection
docs = sc.textFile(data_path,
                   # minPartitions=K  # TODO as said, we should not use this.
                   ).cache()

# Random K partitions.
docs.repartition(numPartitions=K)

# Force data loading
print("Data found: ", docs.count(), " row(s).")


# 2. MapReduce Word count
# Improved Word count 1
def word_count1(sc, docs):
    def f1(document):
        pairs_dict = {}
        for word in document.split(' '):
            if word not in pairs_dict.keys():
                pairs_dict[word] = 1
            else:
                pairs_dict[word] += 1
        return [(key, pairs_dict[key]) for key in pairs_dict.keys()]

    def f2(pair):
        word, occurrences = pair[0], list(pair[1])
        sum_o = 0
        for o in occurrences:
            sum_o += o
        return word, sum_o

    # flatMap(f1) is the map phase
    # groupByKey().map(f2) is the reduce phase
    word_count_pairs = docs.flatMap(f1).groupByKey().map(f2)
    counts = word_count_pairs.reduceByKey().collect()  # TODO like this?
    # data must be small enough to fit in driver's memory
    return counts

def word_count2_random(sc, docs):
    pass

def word_count2_partitioned(sc, docs):
    pass

try:
    import cProfile as profile
except ImportError:
    import profile

profile.run("word_count1(sc, docs)", "G24HM2_word_count1_stats")
profile.run("word_count2_random(sc, docs)", "G24HM2_word_count2_random_stats")
profile.run("word_count2_partitioned(sc, docs)", "G24HM2_word_count2_partitioned_stats")

# Keep open the web interface provided
input("Press [RETURN] to end the program.")
