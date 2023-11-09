import csv

user_item_dict = {}

with open('D:/neural_graph_collaborative_filtering_NGCF/Data/MovieLens/rating.csv', mode='r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  
    for row in csv_reader:
        user_id = row[0] 
        item_id = row [1] 
        user_id = int(user_id)  
        item_id = int(item_id)  
        
        if user_id in user_item_dict:
            if user_id < 30000 and item_id < 60000:
                user_item_dict[user_id].append(item_id)
        else:
            if user_id < 30000 and item_id < 60000:
                user_item_dict[user_id] = [item_id]
    
# print(user_item_dict)
# Tạo file train.txt và test.txt
train_data = {}
test_data = {}
for user_id, item_list in user_item_dict.items():
    split_index = int(0.8 * len(item_list))
    train_data[user_id] = item_list[:split_index]
    test_data[user_id] = item_list[split_index:]

# Ghi train_data vào file train.txt
with open('D:/neural_graph_collaborative_filtering_NGCF/Data/MovieLens/train.txt', 'w') as file:
    for user_id, item_list in train_data.items():
        line = str(user_id) + " " + " ".join(map(str, item_list)) + "\n"
        file.write(line)

# Ghi test_data vào file test.txt
with open('D:/neural_graph_collaborative_filtering_NGCF/Data/MovieLens/test.txt', 'w') as file:
    for user_id, item_list in test_data.items():
        line = str(user_id) + " " + " ".join(map(str, item_list)) + "\n"
        file.write(line)
