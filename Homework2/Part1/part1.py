from pyspark import SparkContext, SparkConf
from pyspark.ml.feature import StopWordsRemover

conf = SparkConf().setAppName("WordCount").setMaster("local")
sc = SparkContext(conf=conf)

# Read input text files
file1_lines = sc.textFile("Part1/the_prophet.txt")
file2_lines = sc.textFile("Part1/war_and_peace.txt")

# Process first book
words_file1 = file1_lines.flatMap(lambda line: line.lower().split())
stop_words = set(StopWordsRemover.loadDefaultStopWords("english"))
filtered_words_file1 = words_file1.filter(lambda word: word not in stop_words)
word_counts_file1 = filtered_words_file1.map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b)

# Process second book
words_file2 = file2_lines.flatMap(lambda line: line.lower().split())
filtered_words_file2 = words_file2.filter(lambda word: word not in stop_words)
word_counts_file2 = filtered_words_file2.map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b)

# Combine word counts from both books
combined_word_counts = word_counts_file1.union(word_counts_file2).reduceByKey(lambda a, b: a + b)

# Sort the combined word counts by count in descending order
sorted_word_counts = combined_word_counts.sortBy(lambda x: x[1], ascending=False)

sorted_word_counts_list = sorted_word_counts.collect()

# Write sorted word counts to a single output text file
with open("Part1/output_1.txt", "w") as f:
    for word, count in sorted_word_counts_list:
        f.write(f"{word}: {count}\n")

input("Press Enter to terminate:")

sc.stop()
