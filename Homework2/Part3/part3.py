from pyspark import SparkContext, SparkConf

conf = SparkConf().setAppName("PageRank").setMaster("local")
sc = SparkContext(conf=conf)

# Function to parse line and extract page ID and links
def parse_line(line):
    parts = line.split(':')
    page_id = int(parts[0])
    links_str = parts[1].strip()[1:-1]  # Remove brackets and leading/trailing spaces
    links = [int(link.strip()) for link in links_str.split(',')] if links_str else []
    return page_id, links

# Read the input file
input_file = "Part3/question3.txt"
lines = sc.textFile(input_file)

# Parse lines and create RDD of (page_id, links) pairs
page_links = lines.map(parse_line)

# Initialize PageRank scores
num_pages = page_links.count()
page_ranks = page_links.map(lambda x: (x[0], 1.0 / num_pages))

# We are running 10 iterations PageRank here
num_iterations = 10

# PageRank algorithm
for i in range(num_iterations):
    joined = page_links.join(page_ranks)
    
    # Calculate contributions to each page's PageRank
    contributions = joined.flatMap(lambda x: [(dest, x[1][1] / len(x[1][0])) for dest in x[1][0]])
    
    # Aggregate the contributions for each page
    page_ranks = contributions.reduceByKey(lambda x, y: x + y) \
                             .mapValues(lambda rank: 0.15 / num_pages + 0.85 * rank)

# Output the final PageRank scores
page_ranks_sorted = page_ranks.sortBy(lambda x: x[1], ascending=False).collect()

with open("Part3/output_3.txt", "w") as f:
    for page_id, page_rank in page_ranks_sorted:
        f.write(f"Page ID: {page_id}, PageRank: {page_rank}\n")

input("Press Enter to terminate:")

sc.stop()
