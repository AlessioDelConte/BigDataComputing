from pyspark import SparkContext, SparkConf, rdd, resultiterable
import sys
import os
import time
import random as rand
from collections import Counter


# 2. MapReduce Word count
# Improved Word count 1
def word_count1(docs):
    def f1(document):
        # We replaced the dictionary creation with the faster Counter implementation
        pairs_dict = Counter(document.split(' '))
        return [(key, pairs_dict[key]) for key in pairs_dict.keys()]

    # flatMap(f1) is the map phase
    # groupByKey().map(f2) has been replaced by reduceByKey, which also computes the shuffle
    word_count_pairs = docs.flatMap(f1).reduceByKey(lambda x, y: x + y)
    # data must be small enough to fit in driver's memory
    return word_count_pairs


def word_count2_random(docs, K):
    def f1(document):
        pairs_dict = Counter(document.split(' '))
        return [(key, pairs_dict[key]) for key in pairs_dict.keys()]

    def f3(grouped_key_value_list):
        pairs_dict = {}                             # Dictionary (key=w, value=C(w, x))
        # Iterate through a single partition with key x = [0, k)
        # We extract the only element in grouped_key_value_list (a partition)
        for word, occ in list(grouped_key_value_list)[0]:
            if word not in pairs_dict.keys():
                pairs_dict[word] = occ
            else:
                pairs_dict[word] += occ
        return [(key, pairs_dict[key]) for key in pairs_dict.keys()]

    word_count_pairs = (
        docs
        .flatMap(f1)                                # Map_round_1: Produce intermediate pairs (w, Ci(w))
        .groupBy(lambda x: rand.randrange(0, K))    # Map_round_1: Produce intermediate pairs (x, [(w, Ci(w))])
        .groupByKey().flatMapValues(f3).values()    # Reduce_round_1: Produce intermediate pairs (w, c(w, x))
        .reduceByKey(lambda x, y: x + y))           # Reduce_round_2: Gather all pairs (w, c(w, x)) and sum up all the
                                                    # partial sums for every word and produce (w, c(w))
    return word_count_pairs


def word_count2_partitioned(docs):
    def f1(documents):
        pairs_dict = {}                             # dictionary (key=w, value=c(w)) for the current partition
        for document in documents:                  # we iterate through n/k documents for k partitions
            for word in document.split(' '):
                if word not in pairs_dict.keys():
                    pairs_dict[word] = 1
                else:
                    pairs_dict[word] += 1
        return [(key, pairs_dict[key]) for key in pairs_dict.keys()]

    # Instead of mapping each single document using the flatMap function we use the mapPartitions function to
    # exploit the existing subdivision of the documents in K partitions to obtain all the pairs (w, c(w, x))),
    # where x is a number in [0, K), exploiting this feature we don't need to perform data shuffling leading to a
    # performance increase. We then gather all this pairs and we reduce them summing up all the occurrences for
    # the same word.
    word_count_pairs = (docs
                        .mapPartitions(f1)
                        .reduceByKey(lambda x, y: x + y))

    return word_count_pairs


def G24HM2():
    # 1. Data loading
    # Import the Dataset
    docs_txt = []

    assert len(sys.argv) == 3, "Usage: python G24HM2.py <K> <file_name>"

    K = sys.argv[1]
    data_path = sys.argv[2]

    assert K.isdigit(), "K must be an integer"
    assert os.path.isfile(data_path) or os.path.isdir(data_path), "File or folder not found"

    K = int(K)

    # Spark Setup
    conf = SparkConf().setAppName('Word count 1-2').setMaster("local[*]")
    sc = SparkContext(conf=conf)

    # Create a parallel collection, we need to specify minPartitions in order to force the creation of K partitions
    # of the data. The randomness is not assured.
    docs = sc.textFile(data_path,
                       minPartitions=K
                       ).cache()

    # We shuffle data in order to have a good random distribution between K partitions.
    docs.repartition(numPartitions=K)

    # Force data loading
    print("Data found: ", docs.count(), " document(s).")

    # 2. MR Word count algorithms with running times.
    begin_wordcount1 = time.time()
    word_count1(docs).collect()
    print("Elapsed time for Word Count 1: ", time.time() - begin_wordcount1, "second(s).")

    begin_wordcount2_random = time.time()
    word_count2_random(docs, K).collect()
    print("Elapsed time for Word Count 2 random: ", time.time() - begin_wordcount2_random, "second(s).")

    begin_wordcount2_partitioned = time.time()
    word_count = word_count2_partitioned(docs)
    word_count.collect()
    print("Elapsed time for Word Count 2 partitioned: ", time.time() - begin_wordcount2_partitioned, "second(s).")

    # 3. Print average length of distinct words
    average_word_len = word_count.keys().map(lambda x: (x, len(x))).values().mean()
    print("Average word length: ", average_word_len)

    # Keep open the web interface provided
    input("Press [RETURN] to end the program. The web interface is located at: http://localhost:4040/jobs/")


if __name__ == "__main__":
    G24HM2()
