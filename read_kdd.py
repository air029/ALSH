import csv

def read_data(file_name, binary = False):
    """read csv data"""

    dimension = 0
    num_data = 0

    with open(file_name, "r") as csv_file:
        reader = csv.reader(csv_file)

        for line in reader:
            num_data +=1
            for ix in line:
                str_line = str(ix)
                #print(str_line)
                list_split_line1 = str_line.split(" ")
                #print(len(list_split_line1))
                #print(list_split_line1)
                str_last_element = list_split_line1[-1]
                #print(str_last_element)
                list_last_element = str_last_element.split(":")
                #print(list_last_element[0])
                if dimension < int(list_last_element[0]):
                    dimension = int(list_last_element[0])

        #print(num_data)
        #print(dimension)

    datas = [[0 for i in range(dimension)] for j in range(num_data)]
    num = 0

    with open(file_name, "r") as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            for ix in line:
                str_line = str(ix)
                list_split_line1 = str_line.split(" ")
                for ij in range(len(list_split_line1)):
                    temp_element = str(list_split_line1[ij])
                    #print(temp_element)
                    if ":" in temp_element:
                        temp_list = temp_element.split(":")
                        temp_index = int(temp_list[0])
                        temp_value = int(temp_list[-1])
                        #print("index")
                        #print(temp_index)
                        #print("value")
                        #print(temp_value)
                        if binary == False:
                            datas[num][temp_index-1] = temp_value
                        else:
                            datas[num][temp_index - 1] = 1
            num += 1

    return datas

def read_num(file_name):
    """read csv data"""

    num_data = 0
    with open(file_name, "r") as csv_file:
        reader = csv.reader(csv_file)

        for line in reader:
            num_data +=1

    return num_data

def read_dimension(file_name):
    """read csv data"""

    dimension = 0

    with open(file_name, "r") as csv_file:
        reader = csv.reader(csv_file)

        for line in reader:
            for ix in line:
                str_line = str(ix)
                list_split_line1 = str_line.split(" ")
                str_last_element = list_split_line1[-1]
                list_last_element = str_last_element.split(":")
                if dimension < int(list_last_element[0]):
                    dimension = int(list_last_element[0])

    return dimension

def read_queries(file_name, num_queries):
    queries = []
    datas = read_data(file_name)
    for i in range(num_queries):
        queries.append(datas[i])

    return queries



#print(read_num("kddalgebra_sample.csv"))
#datas = read_data("kddalgebra_sample.csv", False)



dimension = read_dimension("kddalgebra_sample.csv")
num_data = read_num("kddalgebra_sample.csv")

"""
print(dimension)
print(num_data)
print(dimension * num_data)
list = []
for i in range(dimension * num_data):
    list.append(0)

"""


#datas = [[0 for i in range(dimension)] for j in range(num_data)]
#num = 0

"""
with open(file_name, "r") as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
        for ix in line:
            str_line = str(ix)
            list_split_line1 = str_line.split(" ")
            for ij in range(len(list_split_line1)):
                temp_element = str(list_split_line1[ij])
                #print(temp_element)
                if ":" in temp_element:
                    temp_list = temp_element.split(":")
                    temp_index = int(temp_list[0])
                    temp_value = int(temp_list[-1])
                    #print("index")
                    #print(temp_index)
                    #print("value")
                    #print(temp_value)
                    if binary == False:
                        datas[num][temp_index-1] = temp_value
                    else:
                        datas[num][temp_index - 1] = 1
        num += 1

"""

