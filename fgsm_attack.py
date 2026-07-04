import torch
import torch.nn as nn
import numpy as np
import os
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# -------------------------------
# CONFIG
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "densenet121_chest_xray.pth"
CSV_PATH = "Data_Entry_2017.csv"
TEST_LIST_PATH = "test_list.txt"
IMAGES_DIR = "images"

IMAGE_SIZE = 224
BATCH_SIZE = 64

DISEASE_CLASSES = [
    'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema',
    'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_thickening',
    'Cardiomegaly', 'Nodule', 'Mass', 'Hernia', 'No Finding'
]

# -------------------------------
# DATASET
# -------------------------------
class ChestXRayDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


def load_test_data(csv_path, images_dir, test_list_path):
    df = pd.read_csv(csv_path)
    
    # Read test image list
    with open(test_list_path, 'r') as f:
        test_images = [line.strip() for line in f.readlines()]
    
    image_paths = []
    labels_list = []

    for img_name in test_images:
        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            continue

        # Find matching row in CSV using image name
        # Extract the image index from filename (e.g., "00000003_000.png" -> "00000003_000")
        img_index = img_name.rsplit('.', 1)[0] + '.png'
        
        matching_rows = df[df['Image Index'] == img_index]
        if len(matching_rows) == 0:
            continue
        
        row = matching_rows.iloc[0]
        findings = str(row['Finding Labels']).split('|')
        findings = [f.strip() for f in findings]

        label_vector = [1 if d in findings else 0 for d in DISEASE_CLASSES]

        image_paths.append(img_path)
        labels_list.append(label_vector)

    image_paths = np.array(image_paths)
    labels_array = np.array(labels_list)

    return image_paths, labels_array


def create_test_loader(test_images, test_labels):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = ChestXRayDataset(test_images, test_labels, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    return loader


# -------------------------------
# MODEL
# -------------------------------
def create_model(num_classes=15):
    model = models.densenet121(pretrained=False)

    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )

    return model


# -------------------------------
# FGSM
# -------------------------------
def fgsm_attack(model, images, labels, epsilon, criterion):
    images = images.clone().detach().to(device)
    labels = labels.to(device)

    images.requires_grad = True

    outputs = model(images)
    loss = criterion(outputs, labels)

    model.zero_grad()
    loss.backward()

    grad_sign = images.grad.data.sign()
    adv_images = images + epsilon * grad_sign

    # clamp in normalized space
    adv_images = torch.clamp(adv_images, -3, 3)

    return adv_images.detach()


# -------------------------------
# INFERENCE
# -------------------------------
def run_inference(model, loader, epsilon=None):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    all_preds, all_labels, all_probs = [], [], []

    for images, labels in tqdm(loader):
        images = images.to(device)
        labels = labels.to(device)

        if epsilon is not None:
            images = fgsm_attack(model, images, labels, epsilon, criterion)

        with torch.no_grad():
            outputs = model(images)
            probs = torch.sigmoid(outputs)

        preds = (probs > 0.5).float()

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    return (
        np.concatenate(all_preds),
        np.concatenate(all_labels),
        np.concatenate(all_probs),
    )


# -------------------------------
# METRICS
# -------------------------------
def compute_metrics(preds, labels, probs):
    hamming = np.mean(preds != labels)
    exact_match = np.mean(np.all(preds == labels, axis=1))

    tp = np.sum((preds == 1) & (labels == 1), axis=0)
    fp = np.sum((preds == 1) & (labels == 0), axis=0)
    fn = np.sum((preds == 0) & (labels == 1), axis=0)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    try:
        auc = roc_auc_score(labels, probs, average='macro')
    except:
        auc = 0.0

    return {
        "hamming_loss": hamming,
        "exact_match": exact_match,
        "f1_macro": np.mean(f1),
        "auc_macro": auc
    }


# -------------------------------
# MAIN
# -------------------------------
def main():
    print("Loading test data...")
    test_images, test_labels = load_test_data(CSV_PATH, IMAGES_DIR, TEST_LIST_PATH)
    test_loader = create_test_loader(test_images, test_labels)

    print("Loading model...")
    model = create_model(len(DISEASE_CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)

    print("\nRunning CLEAN evaluation...")
    clean_preds, clean_labels, clean_probs = run_inference(model, test_loader)
    clean_metrics = compute_metrics(clean_preds, clean_labels, clean_probs)

    print("\nCLEAN RESULTS:")
    print(clean_metrics)

    epsilons = [0.001, 0.005, 0.01, 0.02]

    for eps in epsilons:
        print(f"\nRunning FGSM attack (epsilon={eps})...")

        adv_preds, adv_labels, adv_probs = run_inference(
            model, test_loader, epsilon=eps
        )

        adv_metrics = compute_metrics(adv_preds, adv_labels, adv_probs)

        flip_rate = np.mean(clean_preds != adv_preds)

        print(f"\nFGSM RESULTS (eps={eps}):")
        print(adv_metrics)
        print(f"Flip Rate: {flip_rate:.4f}")


if __name__ == "__main__":
    main()