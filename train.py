import config
import torch
import math
import numpy as np

from torch.optim import Adam
from model import Model
from data import Data

class Train():
    def __init__(self, model):
        self.model = model
        self.params = self.model.parameters()
        self.optimizer = Adam(self.params, lr=config.LR, betas=config.ADAM_BETAS, 
                              eps=config.ADAM_EPS, weight_decay=config.ADAM_WEIGHT_DECAY)

        self.crit = torch.nn.BCELoss()


    def train_one_batch(self, batch):
        user_batch, item_batch, rating_batch = batch
        if config.USE_GPU:
            user_batch = user_batch.cuda()
            item_batch = item_batch.cuda()
            rating_batch = rating_batch.cuda()
        self.optimizer.zero_grad()
        ratings_pred = self.model(user_batch, item_batch)
        loss = self.crit(ratings_pred.view(-1), rating_batch)
        loss.backward()
        self.optimizer.step()
        loss = loss.item()
        return loss

    def train_one_epoch(self, dataset, epoch_id):
        self.model.train()
        print('Start training for epoch ', epoch_id)
        dataset.init_batcher()
        iters = 0
        total_loss = 0
        while True:
            batch = dataset.get_one_batch()
            if not batch:
                break
                
            loss = self.train_one_batch(batch)
            total_loss += loss
            iters += 1
            if iters % 5000 == 0:
                print('Epoch {}, Batch {}, Loss {}'.format(epoch_id, iters, total_loss / iters))

        ndcg = self.evaluate(dataset, epoch_id)
        torch.save(self.model.state_dict(), config.MODEL_PATH)

    def evaluate(self, dataset, epoch_id = 0, top_k = 10):
        self.model.eval()
        print('Start evaluating for epoch ', epoch_id)
        ndcg = 0
        for i, rec in enumerate(dataset.test):
            test_user = rec[0]
            test_item = rec[1]
            negative_items = dataset.negative_samples[int(test_user)]
            eval_users = torch.tensor([ test_user for _ in range(100) ]).long()
            eval_items = torch.tensor( np.append(np.array([test_item]), negative_items) ).long()
            if config.USE_GPU:
                eval_users = eval_users.cuda()
                eval_items = eval_items.cuda()

            ratings = self.model(eval_users, eval_items).cpu()

            # test_item's rating is ratings[0]
            # get the rank for the test_item
            test_rank = np.argwhere((-ratings.T[0]).argsort() == 0)[0,0] + 1

            # dcg = \sum \frac{2^{rel_i}-1}{log_2(i+1)}
            # for negative items, dcg is 0
            # for test item, dcg's numerator is 1
            # idcg is always 1, so ndcg = dcg
            # ndcg is 0 if test_item is not ranked in top_k
            if test_rank <= top_k:
                ndcg += math.log(2) / math.log(1 + test_rank)
            
        result = ndcg / config.N_USER
        print('NDCG', result)
        return result

                

if __name__ == '__main__':
    dataset = Data()
    model = Model().model
    train = Train(model)
    print('Start training...')
    for i in range(1, config.EPOCHS + 1):
        train.train_one_epoch(dataset, i)
