from pyspark import SparkContext, SparkConf
from pyspark.ml.feature import StopWordsRemover

conf = SparkConf().setAppName("WordCount")
sc = SparkContext(conf=conf)

# Read input text files
file1_lines = sc.textFile("the_prophet.txt")
file2_lines = sc.textFile("war_and_peace.txt")

# Combine lines from both files
combined_lines = file1_lines.union(file2_lines)
words = combined_lines.flatMap(lambda line: line.lower().split())

stop_words = set(StopWordsRemover.loadDefaultStopWords("english"))
filtered_words = words.filter(lambda word: word not in stop_words)

word_counts = filtered_words.map(lambda word: (word, 1))
word_counts = word_counts.reduceByKey(lambda a, b: a + b)

# Sort the word counts by count in descending order
sorted_word_counts = word_counts.sortBy(lambda x: x[1], ascending=False)

sorted_word_counts_list = sorted_word_counts.collect()

# Write sorted word counts to a single output text file
with open("output_1.txt", "w") as f:
    for word, count in sorted_word_counts_list:
        f.write(f"{word}: {count}\n")

sc.stop()
