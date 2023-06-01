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
rel_data = []
# keep extrema for normalization
maxima = int_l(data[1][6:-1])
minima = int_l(data[1][6:-1])

for row in data[2:]:
    rel_data.append(int_l(row[6:]))

    # update extrema
    for i, val in enumerate(rel_data[-1][:-1]):
        if val < minima[i]:
            minima[i] = val
        
        if val > maxima[i]:
            maxima[i] = val

# normalize data
for row in rel_data:
    for i, val in enumerate(row[:-1]):
        row[i] = (val - minima[i]) / (maxima[i] - minima[i])

# get mean values (desired values) for each class
classes = {}
for row in data[2:]:
    row = row[6:]
    if row[-1] not in classes:
        classes[row[-1]] = []

    classes[row[-1]].append(int_l(row[:-1]))

class_means = {class_: [0]*len(rel_data[0][:-1]) for class_ in classes}
for class_ in classes:
    for row in classes[class_]:
        for j, col in enumerate(row):
            class_means[class_][j] += col
    
    class_means[class_] = [s/len(classes[class_]) for s in class_means[class_]]

# normalize class means
# this is the same as taking the mean of the normalized data (which would probably be more efficient)
for class_ in class_means:
    class_mean = class_means[class_]
    for i, mean in enumerate(class_mean):
        class_mean[i] = (mean - minima[i]) / (maxima[i] - minima[i])

# write means to csv file
with open("data/class-means.csv", "w") as csvf:
    writer = csv.writer(csvf, delimiter=";")

    for class_ in class_means:
        writer.writerow([class_] + class_means[class_])

"""
# write data to csv file
with open("data/dataset-normalized.csv", "w") as csvf:
    writer = csv.writer(csvf, delimiter=";")

    for row in rel_data:
        writer.writerow(row)
"""
