from tqdm import tqdm

import torch.nn

class LinearEval(nn.Module)

    def __init__(self, hparams, device, encoder, feature_dim=100, num_classes=10, freeze_encoder=True):
        self.encoder = encoder
        if freeze_encoder:
            self.freeze_encoder()
        self.classifier = nn.LinearEval(feature_dim, num_classes)
        self.device = device
        self.trainloader, self.traindst, self.testloader, self.testdst = data_loader.get_dataset(self.hparams)
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=1e-6
        )
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma)

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        return self.classifier(features)

    def unfreeze_encoder(self):
        """Unfreeze the encoder for fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def freeze_encoder(self):
        """Freeze the encoder to train only the classifier."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def train_epoch(self, epoch):
        self.classifier.train()
        total_loss, total_num, train_bar = 0.0, 0, tqdm(self.trainloader)
        for i, (ori_image, pos_1, pos_2, target) in enumerate(train_bar):
            # compute logits and loss
            logits = self.forward(x=ori_img)
            loss = nn.CrossEntropyLoss()(logits, target)
            # do optimizer step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # rest ...
            total_num += self.hparams.batch_size
            total_loss += loss.item() * self.hparams.batch_size
            train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4e}'.format(epoch, self.hparams.num_iters, total_loss / total_num))
        self.scheduler.step()
        metrics = {
            'total_loss':total_loss / total_num,
            'epoch': epoch
        }
        return metrics

    def train(self):
        for epoch in range(self.hparams.num_iters):
            metrics = self.train_epoch(epoch=epoch)
            print(f'Epoch: {epoch+1}, metrics: {json.dumps(metrics, indent=4)}')
