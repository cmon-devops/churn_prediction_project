import pandas as pd

def preprocess_data(input_file, output_file):
    # Load data
    data = pd.read_csv(input_file)

    # Basic preprocessing: Convert categorical data and normalize features
    data['is_churn'] = data['churn_label'].apply(lambda x: 1 if x == 'yes' else 0)
    data.drop(['customer_id', 'churn_label'], axis=1, inplace=True)

    # Normalize numerical features
    for col in data.select_dtypes(include=['int', 'float']).columns:
        data[col] = (data[col] - data[col].mean()) / data[col].std()

    data.to_csv(output_file, index=False)

if __name__ == "__main__":
    preprocess_data('data/churn_data.csv', 'data/processed_data.csv')

