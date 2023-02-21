import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import nn
from torch.nn import functional as F
from transformers import BertForSequenceClassification, AdamW
from transformers import BertTokenizer
from pytorch_pretrained_bert import BertModel
import pandas as pd
import numpy as np
import time, datetime, random, glob, os, sys, joblib, argparse, json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import KFold
from sklearn import metrics

USING_GPU = False
DEVICE = None

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
def train(fold, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 5 == 0:
            print('Train Fold/Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                fold,epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(fold,model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set for fold {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        fold,test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# batch size: 16, 32
# learning rate: 5e-5, 3e-5, 2e-5
# epochs: 2,3,4
def train_bert_model(model, train_dataset, tokenizer, batch_size, epochs=2, learning_rate=2e-5, epsilon=1e-8, extras=False, save_fn=None):

    if USING_GPU:
        print("Using GPU", DEVICE)
        model.cuda(DEVICE)

    total_t0 = time.time()
    optimizer = AdamW(model.parameters(),
                      lr=learning_rate,
                      eps=epsilon
                      )

    # prepare cross validation
    n = 5
    kfold = KFold(n_splits=n, shuffle=True)

    for fold, (train_idx, test_idx) in enumerate(kfold.split(train_dataset)):
        print('------------fold no---------{}----------------------'.format(fold))
        print(train_idx)
        print(test_idx)
        train_subsampler = SubsetRandomSampler(train_idx)
        test_subsampler = SubsetRandomSampler(test_idx)

        trainloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_subsampler)
        testloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=test_subsampler)

        for epoch in range(1, epochs + 1):
            train(fold, model, DEVICE, trainloader, optimizer, epochs)
            test(fold, model, DEVICE, testloader)
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
    #X = df.Text.values # x
    #Y = df['label'] # y_true
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
    #input_ids, attention_masks, labels = prepare_dataset(X, np.zeros(len(X)), tokenizer, max_length=400)
    #dataset = TensorDataset(input_ids, attention_masks, labels)
    ## train_bert_model(model, train_dataset, batch_size, epochs=2, learning_rate=2e-5, epsilon=1e-8, extras=False, save_fn=None):
    train_bert_model(model, df[["Text", "label"]], tokenizer, batch_size=args.batchsize, extras=False)

    #y_pred = np.zeros(len(X))
    #if not args.usegraph:
        #flat_logits = run_bert_model(model, dataset, batch_size=args.batchsize, extras=False)
        #y_pred = np.argmax(flat_logits, axis=1).flatten()
        #print(y_pred)
        
    #df['BERT_label'] = y_pred
    #df.to_csv(args.output)
        
if __name__ == "__main__":
    main()
