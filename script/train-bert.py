import torch
from torch.utils.data import TensorDataset, Subset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch import nn
from torch.nn import functional as F
from transformers import BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer
from pytorch_pretrained_bert import BertModel
import pandas as pd
import numpy as np
import time, datetime, random, glob, os, sys, joblib, argparse, json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import KFold
from sklearn import metrics
from tqdm import tqdm 

USING_GPU = False
DEVICE = None

# Pretrained models dictionary
pretrained_models = {'bert': "bert-base-uncased"}

def format_time(seconds):
    seconds_round = int(round((seconds)))
    return str(datetime.timedelta(seconds=seconds_round)) # hh:mm:ss
    
def prepare_dataset(sentences, labels, tokenizer, max_length=100):
    input_ids = []
    attention_masks = []
    for sent in sentences:
        # print(sent)
        try:
            encoded_dict = tokenizer.encode_plus(
                                sent,
                                add_special_tokens = True,
                                max_length = max_length,
                                truncation=True,
                                pad_to_max_length = True,
                                return_attention_mask = True,
                                return_tensors = 'pt'
                           )
        except:
            print("some tweet sent is not correct")
            print(sent)
            exit(0)

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    return input_ids, attention_masks, labels

'''
How to combine pytorch dataloader and k-fold
'''
def train(fold, model, device, train_loader, optimizer, scheduler, loss_func, epoch):
    print('Training...')

    # Measure how long the training epoch takes.
    #t0 = time.time()

    # Reset the total loss for this epoch.
    total_train_loss = 0
    model.train()
    # For each batch of training data...
    for step, batch in enumerate(train_loader):
        model.zero_grad()

        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_loader), elapsed))
        
        b_input_ids = batch[0].to(DEVICE)
        b_input_mask = batch[1].to(DEVICE)
        b_labels = batch[2].to(DEVICE)
        #print(b_labels)
        # https://stackoverflow.com/questions/70548318/bertforsequenceclassification-target-size-torch-size1-16-must-be-the-same
        b_labels = torch.nn.functional.one_hot(b_labels.to(torch.int64), 3)
        #print(b_labels)
        #print(b_labels.shape)

        loss, logits, hidden_states = model(b_input_ids,
                                            token_type_ids=None,
                                            attention_mask=b_input_mask,
                                            labels=b_labels)

        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        del loss, logits, hidden_states

    avg_train_loss = total_train_loss / len(train_loader)

    training_time = format_time(time.time() - t0)
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(training_time))

def test(fold, model, device, test_loader, test_data_len, Y):
    print("Running Validation...")

    model.eval()
    predictions, true_labels = [], []
    for batch in tqdm(test_loader, total=test_data_len):


        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(batch[0].to(DEVICE), token_type_ids=None,
                            attention_mask=batch[1].to(DEVICE))
            b_proba = outputs[0]

            proba = b_proba.detach().cpu().numpy()
            label_ids = batch[2].numpy()

            predictions.append(proba)
            true_labels.append(label_ids)
    print(predictions)
    print(true_labels)

# batch size: 16, 32
# learning rate: 5e-5, 3e-5, 2e-5
# epochs: 2,3,4
def train_bert_model(model, dataset, Y, batch_size, epochs=3, learning_rate=1e-5, epsilon=1e-8, save_fn=None):

    if USING_GPU:
        print("Using GPU", DEVICE)
        model.cuda(DEVICE)

    # prepare cross validation
    n = 5
    kfold = KFold(n_splits=n, shuffle=True)
# for each fold
    for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
        print('------------fold no---------{}----------------------'.format(fold))
        print(train_idx)
        print(test_idx)
        train_tensor = Subset(dataset, train_idx)
        test_tensor = Subset(dataset, test_idx)
        test_data_len = len(test_idx)
        #train_subsampler = SubsetRandomSampler(train_idx)
        #test_subsampler = SubsetRandomSampler(test_idx)

        trainloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=RandomSampler(train_tensor))
        testloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SequentialSampler(test_tensor))
        total_steps = len(trainloader) * epochs
        optimizer = AdamW(model.parameters(),
                          lr=learning_rate,
                          eps=epsilon
                          )
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,  # Default value in run_glue.py
                                                    num_training_steps=total_steps)
        loss_func = torch.nn.CrossEntropyLoss()
        for epoch_i in range(0, epochs):
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            train(fold, model, DEVICE, trainloader, optimizer, scheduler, loss_func, epochs)
            #test(fold, model, DEVICE, testloader, test_data_len, Y)
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, choices=list(range(8)))
    parser.add_argument('--input', type=str, default='..', help='path to the raw input text directory')
    parser.add_argument('--output', type=str, default = '../classified-bert/')
    parser.add_argument('--seed', type=int, default=0, help='random seed for replicability')
    parser.add_argument('--usegraph', action='store_true')
    parser.add_argument('--batchsize', type=int, default=8)
    #parser.add_argument('--logfile', type=str)
    parser.add_argument('--trainedBertModel', type=str, help='values should be from bert, ')
    args = parser.parse_args()

    if args.output.endswith(".out"):
        print(f"ignore the log file: {args.output}")
        exit(0)

    if os.path.exists(args.output):
        print("{} exists and we pass it".format(args.output))
        exit(0)
    # ==== parallel running by having the same log file ====
    # # # define the log and make the parallel
    '''
    LOG_FILE = args.logfile #
    log_files = []
    with open(LOG_FILE) as f:
        for line in f:
            log_files.append(line)
    if args.output+"\n" in log_files:
        print(args.output, "is in the processing and skip it")
        exit(0)
    else:
        print(args.output, "is not in the processing and conduct it")
        with open(LOG_FILE, 'a') as f:
            f.write(args.output+"\n")
    '''
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    global USING_GPU
    global DEVICE
    if torch.cuda.is_available():        
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(args.gpu))
        DEVICE = torch.device("cuda:%s"%args.gpu)
        USING_GPU = True
    else:
        print('No GPU available, using the CPU instead.')
        DEVICE = torch.device("cpu")
        USING_GPU = False
    
    tag=""
    
    df = pd.read_csv(args.input)
    df = df[df['Text'].notna()]
    X = df.Text.values # x
    Y = list(df['label']) # y_true
    #print(Y)
    # RuntimeError: Error(s) in loading state_dict for BertForSequenceClassification:
    # size mismatch for classifier.weight: copying a param with shape torch.Size([3, 768]) from checkpoint, the shape in current model is torch.Size([2, 768]).
    # size mismatch for classifier.bias: copying a param with shape torch.Size([3]) from checkpoint, the shape in current model is torch.Size([2]).
    #
    if args.trainedBertModel not in list(pretrained_models.keys()):
        parser.print_help()
        sys.exit(0)

    used_bert_model = pretrained_models[args.trainedBertModel]
    model = BertForSequenceClassification.from_pretrained(used_bert_model, num_labels = 3)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    input_ids, attention_masks, labels = prepare_dataset(X, np.zeros(len(X)), tokenizer, max_length=400)
    dataset = TensorDataset(input_ids, attention_masks, labels)
    ## train_bert_model(model, train_dataset, batch_size, epochs=2, learning_rate=2e-5, epsilon=1e-8, extras=False, save_fn=None):
    train_bert_model(model, dataset, Y, batch_size=args.batchsize)

    #y_pred = np.zeros(len(X))
    #if not args.usegraph:
        #flat_logits = run_bert_model(model, dataset, batch_size=args.batchsize, extras=False)
        #y_pred = np.argmax(flat_logits, axis=1).flatten()
        #print(y_pred)
        
    #df['BERT_label'] = y_pred
    #df.to_csv(args.output)
        
if __name__ == "__main__":
    main()
