"""
Preprocesses the initial data
Consists of:
-> Keeping only relevant data (sensor values and class)
-> Normalizing data to [0, 1]
Creates file "data/dataset-normalized.csv"
"""
import csv

# read data from csv file
data = []
with open("data/dataset-HAR-PUC-Rio.csv", "r") as csvf:
    reader = csv.reader(csvf, delimiter=";")

    for row in reader:
        data.append(row)

def int_l(l):
    """
    Return a list of integers from a list of strings
    Values that cannot be converted to integers are left as strings
    """
    new_l = []

    for val in l:
        try:
            new_l.append(int(val))
        except ValueError:
            new_l.append(val)

    return new_l

# keep only relevant data
rel_data = [int_l(data[1][6:])]
# keep extrema for normalization
maxima = int_l(data[1][6:-1])
minima = int_l(data[1][6:-1])
# get classes from data
class_means = {}

for row in data[2:]:
    rel_data.append(int_l(row[6:]))
    if row[-1] not in class_means:
        class_means[row[-1]] = [0]*len(minima)

    # update extrema
    for i, val in enumerate(rel_data[-1][:-1]):
        if val < minima[i]:
            minima[i] = val
        
        if val > maxima[i]:
            maxima[i] = val

# calculate means per column
for row in rel_data:
    for i, column in enumerate(row[:-1]):
        class_means[row[-1]][i] += column

class_means_normalized = {}
for class_ in class_means:
    for i in range(len(class_means[class_])):
        class_means[class_][i] /= len([r for r in rel_data if r[-1] == class_])
    
    class_means_normalized[class_] = [(mean - minima[i])/(maxima[i] - minima[i]) for i, mean in enumerate(class_means[class_])]

# write means to csv file
with open("data/class-means.csv", "w") as csvf:
    writer = csv.writer(csvf, delimiter=";")

    for class_ in class_means:
        writer.writerow([class_] + class_means[class_])

# write normalized means to csv file
with open("data/class-means-normalized.csv", "w") as csvf:
    writer = csv.writer(csvf, delimiter=";")

    for class_ in class_means_normalized:
        writer.writerow([class_] + class_means_normalized[class_])

# write extrema to csv file
with open("data/normalization-extrema.csv", "w") as csvf:
    writer = csv.writer(csvf, delimiter=";")

    writer.writerow(["minima"] + minima)
    writer.writerow(["maxima"] + maxima)

"""
# write data to csv file
with open("data/dataset-normalized.csv", "w") as csvf:
    writer = csv.writer(csvf, delimiter=";")

    for row in rel_data:
        writer.writerow(row)
"""
