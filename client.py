# client.py
import os

import torch
import flwr as fl
from flwr.common import Context
from torch.utils.data import DataLoader
from models.unet import UNet
from data.dataset import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_model() -> UNet:
    """U-Netモデルを作成する"""
    return UNet(n_channels=3, n_classes=1).to(device)


def create_data_loaders(client_id: int, iid: bool):
    """データローダーを作成する"""
    trainset, testset = load_data(client_id, iid)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
    # テストデータローダーでシャッフルを有効にする
    testloader = DataLoader(testset, batch_size=32, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
    return trainloader, testloader


def get_client_fn(iid: bool):
    """iid フラグを保持する client_fn を返す関数"""

    def client_fn(context: Context):
        client_id = int(context.node_id)
        trainloader, testloader = create_data_loaders(client_id, iid)
        model = create_model()

        class FlowerClient(fl.client.NumPyClient):
            def get_parameters(self, config):
                """モデルパラメータを取得する"""
                return [val.cpu().numpy() for val in model.state_dict().values()]

            def set_parameters(self, parameters):
                """モデルパラメータを設定する"""
                params_dict = zip(model.state_dict().keys(), parameters)
                state_dict = {k: torch.tensor(v).to(device) for k, v in params_dict}
                model.load_state_dict(state_dict, strict=True)

            def fit(self, parameters, config):
                """モデルを学習する"""
                self.set_parameters(parameters)
                model.train()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                criterion = torch.nn.BCEWithLogitsLoss()

                for epoch in range(5):  # 必要に応じてエポック数を増やす
                    for batch in trainloader:
                        if len(batch) >= 2:
                            images, masks = batch[0].to(device), batch[1].to(device)
                            # print(f"epoch: {epoch}, batch size: {images.size()}")
                        else:
                            print(f"Unexpected batch size: {len(batch)}")
                            continue
                        optimizer.zero_grad()
                        outputs = model(images)
                        loss = criterion(outputs, masks)
                        loss.backward()
                        optimizer.step()
                num_examples = len(trainloader.dataset)
                return (
                    self.get_parameters(config={}),
                    num_examples,
                    {"client_id": client_id},
                )

            def evaluate(self, parameters, config):
                """モデルを評価する"""
                self.set_parameters(parameters)
                model.eval()
                criterion = torch.nn.BCEWithLogitsLoss()

                total_loss = 0.0
                total_dice = 0.0
                total_precision = 0.0
                total_recall = 0.0
                total_accuracy = 0.0
                total_iou = 0.0
                num_samples = 0

                with torch.no_grad():
                    for batch in testloader:
                        if len(batch) >= 2:
                            images, masks = batch[0].to(device), batch[1].to(device)
                        else:
                            print(f"Unexpected batch size: {len(batch)}")
                            continue

                        outputs = model(images)
                        loss = criterion(outputs, masks)
                        total_loss += loss.item() * images.size(0)

                        # シグモイドを適用して確率に変換
                        probs = torch.sigmoid(outputs)
                        preds = (probs > 0.5).float()

                        # マスクをバイナリに変換
                        masks = (masks > 0.5).float()

                        batch_size = images.size(0)
                        # 各サンプルごとにメトリクスを計算
                        for i in range(batch_size):
                            pred = preds[i]
                            target = masks[i]

                            dice = self.dice_coeff(pred, target)
                            precision = self.precision(pred, target)
                            recall = self.recall(pred, target)
                            accuracy = self.accuracy(pred, target)
                            iou = self.iou(pred, target)

                            total_dice += dice
                            total_precision += precision
                            total_recall += recall
                            total_accuracy += accuracy
                            total_iou += iou

                            num_samples += 1

                avg_loss = total_loss / num_samples
                avg_dice = total_dice / num_samples
                avg_precision = total_precision / num_samples
                avg_recall = total_recall / num_samples
                avg_accuracy = total_accuracy / num_samples
                avg_iou = total_iou / num_samples

                num_examples = len(testloader.dataset)

                return (
                    float(avg_loss),
                    num_examples,
                    {
                        "dice_score": float(avg_dice),
                        "precision": float(avg_precision),
                        "recall": float(avg_recall),
                        "accuracy": float(avg_accuracy),
                        "IoU": float(avg_iou),
                    },
                )

            def dice_coeff(self, pred, target):
                """Dice係数を計算する"""
                smooth = 1e-6
                intersection = (pred * target).sum()
                union = pred.sum() + target.sum()
                dice = (2.0 * intersection + smooth) / (union + smooth)
                return dice.item()

            def precision(self, pred, target):
                """Precisionを計算する"""
                smooth = 1e-6
                true_positives = (pred * target).sum()
                predicted_positives = pred.sum()
                precision = (true_positives + smooth) / (predicted_positives + smooth)
                return precision.item()

            def recall(self, pred, target):
                """Recallを計算する"""
                smooth = 1e-6
                true_positives = (pred * target).sum()
                actual_positives = target.sum()
                recall = (true_positives + smooth) / (actual_positives + smooth)
                return recall.item()

            def accuracy(self, pred, target):
                """Accuracyを計算する"""
                correct = (pred == target).sum()
                total = torch.numel(pred)
                accuracy = correct.float() / total
                return accuracy.item()

            def iou(self, pred, target):
                """IoUを計算する"""
                smooth = 1e-6
                intersection = (pred * target).sum()
                union = pred.sum() + target.sum() - intersection
                iou = (intersection + smooth) / (union + smooth)
                return iou.item()

        return FlowerClient().to_client()

    return client_fn
