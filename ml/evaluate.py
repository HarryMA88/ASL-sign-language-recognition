import torch

def run_test(model, test_loader, device, model_name):
    import matplotlib.pyplot as plt

    model.load_state_dict(torch.load(f"{model_name}_model.pt", map_location=device, weights_only=True))
    model.eval()

    correct, total = 0, 0
    misclassified = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Track mismatches
            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    misclassified.append((images[i].cpu(), labels[i].item(), preds[i].item()))

    print(f"Test Accuracy: {correct / total:.4f}")

    print(f"\nMisclassified Samples (showing up to 50):")
    for i, (img, true_label, pred_label) in enumerate(misclassified[:50]):
        plt.imshow(img.squeeze(), cmap="gray")
        plt.title(f"True: {true_label} | Pred: {pred_label}")
        plt.axis("off")
        plt.show()
