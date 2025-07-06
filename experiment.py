import os

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    balanced_accuracy_score, roc_curve, auc, precision_recall_curve,
    average_precision_score
)
from torch import optim
from torchvision.utils import save_image
from tqdm import tqdm

from models import BaseVAE
from models.types_ import *


class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params

        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def on_train_epoch_end(self):
        loss = self.trainer.callback_metrics.get('loss')
        val_loss = self.trainer.callback_metrics.get('val_loss')
        if loss is not None:
            print(f"[Epoch {self.current_epoch}] loss: {loss.item():.4f}", end=' ')
            self.train_losses.append(loss.item())
        if val_loss is not None:
            print(f"val_loss: {val_loss.item():.4f}")
        self.val_losses.append(val_loss.item())

    def training_step(self, batch, batch_idx):  #, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        train_loss = self.model.loss_function(*results,
                                              M_N=self.params['kld_weight'],  #al_img.shape[0]/ self.num_train_imgs,
                                              #optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx):  #, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        val_loss = self.model.loss_function(*results,
                                            M_N=1.0,  #real_img.shape[0]/ self.num_val_imgs,
                                            #optimizer_idx = optimizer_idx,
                                            batch_idx=batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def on_validation_end(self) -> None:
        #self.sample_images()
        self.sample_images_next_to_origs()

    def sample_images(self):
        # Get sample reconstruction image            
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

        #         test_input, test_label = batch
        recons = self.model.generate(test_input, labels=test_label)
        vutils.save_image(recons.data,
                          os.path.join(self.logger.log_dir,
                                       "Reconstructions",
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)

        try:
            samples = self.model.sample(144,
                                        self.curr_device,
                                        labels=test_label)
            vutils.save_image(samples.cpu().data,
                              os.path.join(self.logger.log_dir,
                                           "Samples",
                                           f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=12)
        except Warning:
            pass


    def sample_images_next_to_origs(self, num_pairs_per_row=5):

        # Get sample batch
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

        # Generate reconstructions
        recons = self.model.generate(test_input, labels=test_label)

        # Limit to a multiple of num_pairs_per_row for clean layout
        max_pairs = (test_input.size(0) // num_pairs_per_row) * num_pairs_per_row
        test_input = test_input[:max_pairs]
        recons = recons[:max_pairs]

        # Interleave original and reconstruction: [orig1, recon1, orig2, recon2, ...]
        interleaved = torch.stack([test_input, recons], dim=1).view(-1, *test_input.shape[1:])

        # Save directory
        recon_dir = os.path.join(self.logger.log_dir, "Reconstructions")
        os.makedirs(recon_dir, exist_ok=True)

        # Save image grid with num_pairs_per_row * 2 images per row
        # (2 images per pair: original and reconstruction)
        vutils.save_image(
            interleaved.data,
            os.path.join(
                recon_dir,
                f"recons_grid_{self.logger.name}_Epoch_{self.current_epoch}.png"
            ),
            nrow=num_pairs_per_row * 2,  # two images per pair
            normalize=True,
            pad_value=1.0
        )

    def visualize_latent_space(self, dataloader, method="tsne", max_samples=1000, title="Latent Space"):
        self.model.eval()
        latents = []
        labels = []

        device = next(self.model.parameters()).device

        with torch.no_grad():
            for batch in dataloader:
                x, y = batch
                x = x.to(device)
                mu, logvar = self.model.encode(x)
                z = self.model.reparameterize(mu, logvar)
                latents.append(z.cpu())
                labels.append(y)
                if len(latents) * x.size(0) >= max_samples:
                    break

        latents = torch.cat(latents).numpy()
        labels = torch.cat(labels).numpy() if isinstance(labels[0], torch.Tensor) else np.array(labels)

        # Reduce to 2D
        if method == "tsne":
            reducer = TSNE(n_components=2)
        elif method == "pca":
            reducer = PCA(n_components=2)
        else:
            raise ValueError("Unsupported method")

        reduced = reducer.fit_transform(latents)

        # Plot
        plt.figure(figsize=(8, 6))
        if labels.ndim == 1:
            scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', s=5)
            plt.legend(*scatter.legend_elements(), title="Label")
        else:
            plt.scatter(reduced[:, 0], reduced[:, 1], s=5)
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{title.replace(' ', '_').lower()}.png")
        plt.show()


    def get_error_vecs_meaned_per_pixel(self, dataloader):
        model = self.model.eval()
        device = next(model.parameters()).device

        errors = []
        with torch.no_grad():
            for x, _ in tqdm(dataloader, desc="Reconstructing images"):
                x = x.to(device)
                recon = model.generate(x)
                err = F.mse_loss(recon, x, reduction="none")
                err = err.flatten(1).mean(dim=1)
                errors.extend(err.cpu().numpy())

        return np.array(errors)

    def get_error_vecs_per_channel(self, dataloader):
        model = self.model.eval()
        device = next(model.parameters()).device

        error_vecs = []
        labels = []

        with torch.no_grad():
            for x, lbl in tqdm(dataloader, desc="Reconstructing images"):
                x = x.to(device)
                recon = model.generate(x)
                err = F.mse_loss(recon, x, reduction="none")  # [B, C, H, W]
                err = err.view(err.size(0), err.size(1), -1).mean(dim=2)  # [B, C] → mean per channel

                error_vecs.append(err.cpu().numpy())
                labels.append(lbl.cpu().numpy())

        error_vecs = np.concatenate(error_vecs, axis=0)
        labels = np.concatenate(labels, axis=0)

        return error_vecs, labels


    def classify_cracked_images_with_k_sigma_rule(self,
                                                  dataloaders,  # [val_loader, test_loader]
                                                  result_dir: str = "./logs/",
                                                  k: float = 3.0):
        """
        1. Compute threshold = μ + k·σ of reconstruction error on the *validation* set
        2. Classify each image in the *test* set with that threshold
        3. Save images into result_dir/{crack|normal}/
        4. Print accuracy, precision, recall, F1 score, confusion matrix
        5. Plot ROC and Precision-Recall curves
        """
        val_loader, test_loader = dataloaders
        model = self.model.eval()
        device = next(model.parameters()).device
        os.makedirs(result_dir, exist_ok=True)

        val_errors = self.get_error_vecs_meaned_per_pixel(val_loader)

        mu, sigma = val_errors.mean(), val_errors.std()
        threshold = mu + k * sigma
        print(f"\nThreshold set to: {threshold:.6f}  (μ={mu:.6f}, σ={sigma:.6f}, k={k})")

        # ───── 2) Classify test set ─────
        correct, total = 0, 0
        cracked, uncracked = [], []
        y_true, y_pred, y_scores = [], [], []  # scores = raw reconstruction errors

        with torch.no_grad():
            for idx, (x, lbl) in enumerate(tqdm(test_loader, desc="Testing")):
                x = x.to(device)
                recon = model.generate(x)
                err = F.mse_loss(recon, x, reduction="none")
                err = err.flatten(1).mean(dim=1).cpu().numpy()

                for j, e in enumerate(err):
                    true_lbl = lbl[j].item()
                    pred_lbl = 1 if e > threshold else 0

                    y_true.append(true_lbl)
                    y_pred.append(pred_lbl)
                    y_scores.append(e)

                    total += 1
                    if pred_lbl == true_lbl:
                        correct += 1

                    tag = "crack" if pred_lbl == 1 else "normal"
                    save_dir = os.path.join(result_dir, tag)
                    os.makedirs(save_dir, exist_ok=True)
                    save_image(x[j].cpu(), os.path.join(save_dir, f"{idx:05d}_{j}.png"))

                    (cracked if pred_lbl == 1 else uncracked).append((idx, e))

        # ───── 3) Metrics ─────
        acc = 100 * correct / total
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        print("\n--- k-sigma rule Classification Report ---")
        print(f"\nTotal images         : {total}")
        print(f"Correct              : {correct}  ({acc:.2f} %)")
        print(f"Incorrect            : {total - correct}  ({100 - acc:.2f} %)")
        print(f"Predicted 'crack'    : {len(cracked)}")
        print(f"Predicted 'normal'   : {len(uncracked)}")
        print("\n--- Classification Report ---")
        print(f"Precision            : {precision:.4f}")
        print(f"Recall               : {recall:.4f}")
        print(f"F1 Score             : {f1:.4f}")
        print(f"Balanced Accuracy    : {bal_acc:.4f}")
        print(f"Confusion Matrix     : TP={tp}, FP={fp}, TN={tn}, FN={fn}")

        # ───── 4) ROC & PR Curves ─────
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
        avg_precision = average_precision_score(y_true, y_scores)

        # Plot ROC
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.10f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(os.path.join(result_dir, "roc_curve.png"))
        plt.show()
        plt.close()

        # Plot PR Curve
        plt.figure()
        plt.plot(recall_curve, precision_curve, label=f"AP = {avg_precision:.10f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.savefig(os.path.join(result_dir, "precision_recall_curve.png"))
        plt.show()
        plt.close()

        y_scores = np.array(y_scores)
        y_true = np.array(y_true)

        plt.hist(y_scores[y_true == 1], bins=100, alpha=0.5, label='crack (true 1)')
        plt.hist(y_scores[y_true == 0], bins=100, alpha=0.5, label='normal (true 0)')
        plt.xlabel('Reconstruction error')
        plt.ylabel('Count')
        plt.legend()
        plt.title('Distribution of reconstruction errors')
        plt.savefig(os.path.join(result_dir, "histogram_reconstruction_errors.png"))
        plt.grid(True)
        plt.show()

        return cracked, uncracked, threshold

    def classify_with_fbeta(self, dataloaders, result_dir="./logs/", beta=1.0):
        val_loader, test_loader = dataloaders
        model = self.model.eval()
        device = next(model.parameters()).device
        os.makedirs(result_dir, exist_ok=True)

        # ───── Step 1: MLE fit from val set ─────
        val_error_vecs, _ = self.get_error_vecs_per_channel(val_loader)

        print("error vec shape in classify_with_fbeta: " + str(val_error_vecs.shape))

        mu, sigma = self.calculate_mle_mu_sigma(val_error_vecs)

        # ───── Step 2: Get test errors ─────
        # test_error_vecs = []
        # y_true = []
        # x_all = []
        #
        # with torch.no_grad():
        #     for x, lbl in tqdm(test_loader, desc="Testing"):
        #         x = x.to(device)
        #         recon = model.generate(x)
        #         err = F.mse_loss(recon, x, reduction="none")
        #         err = err.flatten(1)  # shape: [B, D]
        #
        #         test_error_vecs.extend(err.cpu().numpy())
        #         y_true.extend(lbl.cpu().numpy())
        #         x_all.extend(x.cpu())  # for saving images later
        #
        # test_error_vecs = np.array(test_error_vecs)

        x_all = []
        with torch.no_grad():
            for x, lbl in tqdm(test_loader, desc="Scuffed workaround to save images"):
                x_all.extend(x.cpu())

        test_error_vecs, y_true = self.get_error_vecs_per_channel(test_loader)

        # ───── Step 3: Compute anomaly scores ─────
        scores = self.compute_anomaly_score(test_error_vecs, mu, sigma)

        # ───── Step 4: Find threshold (optional) ─────
        threshold, best_fbeta = self.find_optimal_threshold(scores, y_true, beta=beta)
        print(f"Best threshold from PR curve (F{beta}): {threshold:.4f}")

        # ───── Step 5: Classification and metrics ─────
        y_pred = (scores > threshold).astype(int)

        cracked, uncracked = [], []
        for i, (img, score, pred, true_lbl) in enumerate(zip(x_all, scores, y_pred, y_true)):
            tag = "crack" if pred == 1 else "normal"
            os.makedirs(os.path.join(result_dir, tag), exist_ok=True)
            save_image(img, os.path.join(result_dir, tag, f"{i:05d}.png"))

            (cracked if pred == 1 else uncracked).append((i, score))

        # ───── Metrics ─────
        correct = (y_pred == y_true).sum()
        acc = 100 * (y_pred == y_true).sum() / len(y_true)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        print("\n--- F-beta-MLE Classification Report ---")
        print(f"\nTotal images         : {str(len(y_true))}")
        print(f"Correct              : {correct}  ({acc:.2f} %)")
        print(f"Incorrect            : {len(y_true) - correct}  ({100 - acc:.2f} %)")
        print(f"Predicted 'crack'    : {len(cracked)}")
        print(f"Predicted 'normal'   : {len(uncracked)}")
        print("\n--- Classification Report ---")
        print(f"Precision         : {precision:.4f}")
        print(f"Recall            : {recall:.4f}")
        print(f"F1 Score          : {f1:.4f}")
        print(f"Balanced Accuracy : {bal_acc:.4f}")
        print(f"Confusion Matrix  : TP={tp}, FP={fp}, TN={tn}, FN={fn}")

        return cracked, uncracked, threshold


    def calculate_mle_mu_sigma(self, error_vecs):
        mu = np.mean(error_vecs, axis=0)
        sigma = None

        print("error vec shape in calculate_mle_mu_sigma: " + str(error_vecs.shape))

        if error_vecs.shape[1] == 1:
            sigma = np.var(error_vecs)  # variance for 1D vectors
        else:
            sigma = np.cov(error_vecs, rowvar=False)  # Covariance matrix for mD vectors

        print("mu: " + str(mu))
        print("sigma: " + str(sigma))

        return mu, sigma

    def compute_anomaly_score(self, error_vecs, mu, sigma):
        print("error_vecs shape in compute_anomaly_score: " + str(error_vecs.shape))
        scores_of_seq = None

        if error_vecs.shape[1] == 1:
            # scores_of_window.append(np.square(data_point - mu) / sigma) #z_score for univariate data -->doesnt work
            # scores_of_seq = np.sqrt(np.square(np.subtract(error_vecs, mu)))
            # Z-score for univariate data
            scores_of_seq = np.abs(error_vecs - mu) / sigma
        else:
            # Mahalanobis distance for multivariate data
            inv_cov_matr = np.linalg.inv(sigma)
            diff = error_vecs - mu
            scores_of_seq = np.einsum('ij,jk,ik->i', diff, inv_cov_matr, diff)

        return scores_of_seq

    def find_optimal_threshold(self, anomaly_scores, true_labels, beta):
        anomaly_scores = anomaly_scores.flatten()
        true_labels = np.squeeze(true_labels).flatten()

        print("anomaly_scores shape: " + str(anomaly_scores.shape))
        print("true_labels shape: " + str(true_labels.shape))

        precision, recall, thresholds = precision_recall_curve(y_true=true_labels, y_score=anomaly_scores)
        fbeta_scores = (1 + beta ** 2) * (precision * recall) / ((beta ** 2) * precision + recall)
        fbeta_scores = np.nan_to_num(fbeta_scores, nan=-np.inf, posinf=-np.inf)

        print("fbeta scores: \n" + str(fbeta_scores))
        print("best fbeta: " + str(np.max(fbeta_scores)))
        best_index = np.argmax(fbeta_scores)
        if best_index >= len(thresholds):
            best_index = len(thresholds) - 1

        print("Thresholds: " + str(thresholds))
        print("Best threshold: " + str(thresholds[best_index]))
        print(str(thresholds))
        best_threshold = thresholds[best_index]
        best_fbeta = fbeta_scores[best_index]

        # plt.figure()
        # plt.plot(thresholds, precision[:-1], 'b-', label='Precision')
        # plt.plot(thresholds, recall[:-1], 'g-', label='Recall')
        # plt.xlabel('Threshold')
        # plt.ylabel('Precision/Recall')
        # plt.legend(loc='best')
        # plt.title('Precision-Recall Curve')
        # plt.show()

        return best_threshold, best_fbeta

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model, self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma=self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma=self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims
