import torch
from NGCF import NGCF
from utility.helper import *
from utility.metrics import *
from utility.batch_test import *
import multiprocessing
import csv

cores = multiprocessing.cpu_count() // 2
file_encoding = 'utf-8'

def predict_for_user(user_id, model, K, drop_flag):
    user_items = []
    try:
        training_items = set(data_generator.train_items[user_id])
        candidate_items = list(filter(lambda item_id: item_id not in training_items, range(ITEM_NUM)))
    except Exception:
        candidate_items = range(ITEM_NUM)

    user_items.append((user_id, candidate_items))

    top_items = []
    with torch.no_grad():
        for user_id, candidate_items in user_items:
            user_tensor = torch.LongTensor([user_id]).to(args.device)
            item_tensor = torch.LongTensor(candidate_items).to(args.device)

            user_embedding, item_embeddings, _ = model(user_tensor, item_tensor, None, drop_flag=drop_flag)
            scores = torch.matmul(user_embedding, item_embeddings.t())

            _, top_item_indices = torch.topk(scores.squeeze(), k=min(K, len(candidate_items)))
            top_items.append((user_id, [candidate_items[i] for i in top_item_indices]))
    
    return top_items
def predict(model, users_to_predict, K, drop_flag=False):
    predictions = {}
    pool = multiprocessing.Pool(cores)
    top_items_batch = predict_for_user(users_to_predict, model, K, drop_flag)

    for user_id, top_items in top_items_batch:
        predictions[user_id] = top_items
    return predictions
if __name__ == '__main__':
    args.device = torch.device('cuda:' + str(args.gpu_id))

    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()

    args.node_dropout = eval(args.node_dropout)
    args.mess_dropout = eval(args.mess_dropout)

    model = NGCF(data_generator.n_users,data_generator.n_items,norm_adj,args).to(args.device)
    model_path = "D:/neural_graph_collaborative_filtering_NGCF/Model_MovieLens/model_MovieLens_1.pth"

    model.load_state_dict(torch.load(model_path))
    model.eval()

    user_id_to_predict = 15
    # users_to_predict = [user_id_to_predict]
    K = 5

    predictions = predict(model, user_id_to_predict, K)
    top_items = predictions[user_id_to_predict]

    print(f"Top {K} predicted items for user {user_id_to_predict}: {top_items}")
    
top_items_info= {}  
with open('D:/neural_graph_collaborative_filtering_NGCF/Data/movielens/movie.csv', mode='r', encoding=file_encoding) as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  
    for row in csv_reader:
        item_id = row [0] 
        title_item = row [1]
        item_id = int(item_id)  
        title_item = str(title_item)  
        
        if item_id in top_items:
            if user_id_to_predict in top_items_info:
                top_items_info[item_id].append(title_item)
            else:
                top_items_info[item_id] = [title_item]
                
sorted_dict = {key: top_items_info[key] for key in top_items if key in top_items_info}           
print(sorted_dict)