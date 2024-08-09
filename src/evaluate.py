import matplotlib.pyplot as plt
import json

def plot_accuracies(results):
    model_names = list(results.keys())
    accuracies = [results[model]['test_accuracy'] for model in model_names]
    plt.figure(figsize=(12, 6))
    plt.bar(model_names, accuracies)
    plt.xlabel('Model')
    plt.ylabel('Test Accuracy')
    plt.title('Model Test Accuracy Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('results/model_accuracy_comparison.png')
    plt.show()

def load_results():
    with open('results/results.json', 'r') as f:
        return json.load(f)

def main():
    results = load_results()
    plot_accuracies(results)

if __name__ == "__main__":
    main()
