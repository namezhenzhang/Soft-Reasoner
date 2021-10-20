import loaddata
from transformers import (AutoModelForMaskedLM, AutoTokenizer,
                          RobertaForSequenceClassification, RobertaModel,
                          RobertaTokenizer)
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torch
torch.nn.BatchNorm2d
from configs import get_args_parser
from optim import get_optimizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
args = get_args_parser()
class rulebert(torch.nn.Module):

    def __init__(self):
        super(rulebert, self).__init__()

        self.roberta = RobertaModel.from_pretrained(
            "roberta-base", return_dict=True)
        self.linear1 = torch.nn.Linear(768, 2)
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)
        self.linear2 = torch.nn.Linear(256, 2)

        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids,attention_mask, labels=None):

        x = self.roberta(input_ids,attention_mask).pooler_output
        x = self.linear1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        # x = self.linear2(x)

        if labels != None:
            loss = self.loss(x, labels)
            return loss
        else:
            return x


model = RobertaForSequenceClassification.from_pretrained("roberta-large",return_dict=True,num_labels=2)
# model = rulebert()
model.cuda()


inputs = loaddata.load_inputs_from_json()
traindataset = loaddata.train_dataset(inputs, 300)
# traindataset.cuda()
traindataloader = DataLoader(traindataset, batch_size=args.per_gpu_train_batch_size, shuffle=True)

# forward pass
optimizer, scheduler = get_optimizer(model, traindataloader)
# optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,)
criterion = torch.nn.CrossEntropyLoss()
global_step = 0
tmp_loss = 0.0
for i in range(args.num_train_epochs):
    with tqdm(total=len(traindataloader)) as t:
        for step,data in enumerate(traindataloader):
            global_step += 1
            # t.set_description(f'epoch:{i}')
            # loss = model(data['input_ids'].cuda(
            # ), data['attention_mask'].cuda(), data['labels'].cuda())
            loss = model(data['input_ids'].cuda(
            ), data['attention_mask'].cuda(), labels=data['labels'].cuda()).loss
            # t.set_postfix(loss=loss.item(), label=data['labels'])
            # t.update(1)

            tmp_loss += loss.item()
            if global_step % args.print_loss_step == 0:
                print(f'{global_step}: {tmp_loss/args.print_loss_step:.6f}')
                tmp_loss = 0.0

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(traindataloader) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

 


