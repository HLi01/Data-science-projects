# %%
#Create a function that decides if a list contains any odd numbers.
#return type: bool
#function name must be: contains_odd
#input parameters: input_list

# %%
def contains_odd(input_list):
    idx = 0
    while idx < len(input_list):
        if input_list[idx] % 2 == 1:
            return True
        idx+=1
    return False

# %%
#Create a function that accepts a list of integers, and returns a list of bool.
#The return list should be a "mask" and indicate whether the list element is odd or not.
#(return should look like this: [True,False,False,.....])
#return type: list
#function name must be: is_odd
#input parameters: input_list

# %%
def is_odd(input_list):
    bool_list=[]
    for item in input_list:
        if item % 2 == 1:
            bool_list.append(True)
        else:
            bool_list.append(False)
    return bool_list

# %%
#Create a function that accpects 2 lists of integers and returns their element wise sum. 
#(return should be a list)
#return type: list
#function name must be: element_wise_sum
#input parameters: input_list_1, input_list_2 

# %%
def element_wise_sum(input_list_1, input_list_2):
    sum_list=[]
    idx=0
    while idx < min(len(input_list_1), len(input_list_2)):
        sum_list.append(input_list_1[idx]+input_list_2[idx])
        idx +=1
    if idx<len(input_list_1):
        for index in range(idx, len(input_list_1)):
            sum_list.append(input_list_1[index])
    elif idx<len(input_list_2):
        for index in range(idx, len(input_list_2)):
            sum_list.append(input_list_2[index])
    return sum_list

# %%
#Create a function that accepts a dictionary and returns its items as a list of tuples
#(return should look like this: [(key,value),(key,value),....])
#return type: list
#function name must be: dict_to_list
#input parameters: input_dict

# %%
def dict_to_list(input_dict):
    list_of_tuples=[]
    for key, value in input_dict.items(): 
        list_of_tuples.append((key,value))
    return list_of_tuples


