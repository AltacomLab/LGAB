from src.dataset.load_datasets import upload_and_clean_csv

def main():
    print(" Upload and clean dataset")
    df = upload_and_clean_csv()
    print("Dataset ready!")
    
 
if __name__ == "__main__":
    main()
