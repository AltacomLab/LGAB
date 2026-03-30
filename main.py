from src.dataset.load_datasets import upload_and_clean_csv

def main():
    print("🔹 Upload and clean dataset")
    df = upload_and_clean_csv()
    print("✅ Dataset ready!")
    
    # TODO: Tiếp tục xử lý dữ liệu, train model, evaluate
    # from src.models.gnn_model import GNNModel
    # from src.training.trainer import train_model
    # from src.evaluation.evaluator import evaluate_model

if __name__ == "__main__":
    main()
