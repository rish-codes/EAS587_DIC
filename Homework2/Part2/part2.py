from pyspark import SparkContext, SparkConf

# Initialize SparkContext
conf = SparkConf().setAppName("Dijkstra").setMaster("local")
sc = SparkContext(conf=conf)

# Function to parse line and extract source, destination, and cost
def parse_line(line):
    parts = line.split(',')
    return int(parts[0]), int(parts[1]), int(parts[2])

# Function for Dijkstra's algorithm
def dijkstra(graph, source):
    # Initialization
    distances = {source: 0}
    visited = set()
    vertices = list(graph.keys())
    
    # Main loop
    while vertices:
        min_vertex = min(vertices, key=lambda v: distances.get(v, float('inf')))
        if min_vertex not in graph:
            break  # Exit loop if no edges for the vertex
        min_distance = distances[min_vertex]
        visited.add(min_vertex)
        vertices.remove(min_vertex)
        for neighbor, cost in graph[min_vertex]:
            if neighbor not in visited:
                new_distance = min_distance + cost
                if new_distance < distances.get(neighbor, float('inf')):
                    distances[neighbor] = new_distance
    
    return distances

# Read text files into RDDs
file1 = sc.textFile("Part2/question2_1.txt")
file2 = sc.textFile("Part2/question2_2.txt")

# Parse lines of both files
parsed_file1 = file1.map(parse_line)
parsed_file2 = file2.map(parse_line)

# Convert RDDs to graph representations
graph_rdd1 = parsed_file1.map(lambda x: (x[0], [(x[1], x[2])])).reduceByKey(lambda x, y: x + y).collectAsMap()
graph_rdd2 = parsed_file2.map(lambda x: (x[0], [(x[1], x[2])])).reduceByKey(lambda x, y: x + y).collectAsMap()

# Run Dijkstra's algorithm on each graph
source_vertex = 0 
result1 = dijkstra(graph_rdd1, source_vertex)
result2 = dijkstra(graph_rdd2, source_vertex)


# Sort results seperately in desc order
sorted_results1 = dict(sorted(result1.items(), key=lambda x: x[1], reverse=True))
sorted_results2 = dict(sorted(result2.items(), key=lambda x: x[1], reverse=True))


# Write sorted results to a single text file
with open("Part2/output_2.txt", "w") as f:
    f.write("Distances from vertex 0 in graph 1: \n")
    for vertex, distance in sorted_results1.items():
        f.write(f"Vertex: {vertex}, Distance: {distance}\n")

    f.write("\n")

    f.write("Distances from vertex 0 in graph 2: \n")
    for vertex, distance in sorted_results2.items():
        f.write(f"Vertex: {vertex}, Distance: {distance}\n")

input("Press Enter to terminate:")

sc.stop()
