import csv 
import random
  
csv_in  = open("house-votes-84.data.csv", "r")
csv_out = open("data.csv", "w")

for line in csv_in:
    field_list = line.split(',')
    for index, elem in enumerate(field_list):
        if elem == '?':
            field_list[index] = str(random.randint(0, 1))
        if elem == 'y':
            field_list[index] = '1'
        if elem == 'n':
            field_list[index] = '0'
        
    new_order = field_list[1:15]
    new_order.append(field_list[0])
    output_line = ','.join(new_order)
    csv_out.write(output_line + "\n")

csv_in.close()
csv_out.close()