import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score


def load_and_clean_data(file_path):
    data = pd.read_csv(file_path)

    # Filter jobs with median salary only (no max or min salary)
    median_only_jobs = data[(data['med_salary_x'].notna()) & (data['max_salary_x'].isna()) & (data['min_salary_x'].isna())]
    max_min_salary_jobs = data[(data['max_salary_x'].notna()) | (data['min_salary_x'].notna())]

    data = data.dropna(subset=['max_salary_x']).reset_index(drop=True)
    data.dropna(axis=1, how='all', inplace=True)

    columns_to_drop = ['industry_id', 'currency_y', 'skill_abr', 'url_x', 'address_x', 'name_x', 'job_posting_url_x',
                       'location_x', 'skills_desc_x', 'posting_domain_x', 'currency_x', 'zip_code_y_x', 'job_posting_url_y',
                       'application_url_y', 'application_type_y', 'posting_domain_y', 'skills_desc_y', 'work_type_y',
                       'inferred_y', 'company_size_y', 'address_y', 'url_y', 'application_url_x', 'description_x_y',
                       'name_y', 'description_y_y', 'zip_code_y_y', 'Unnamed: 0', 'formatted_experience_level_y',
                       'country_y', 'location_y', 'state_y', 'company_name_y', 'max_salary_x', 'min_salary_x',
                       'pay_period_x', 'remote_allowed_x', 'closed_time_x', 'sponsored_x', 'compensation_type_x',
                       'type_x', 'city_x', 'employee_count_x', 'follower_count_x', 'industry_name_x', 'skill_name_x',
                       'formatted_work_type_x', 'type_x', 'title_y']

    data.drop(columns_to_drop, axis=1, inplace=True)

    new_column_names = {
        'company_name_x': 'Company Name',
        'title_x': 'Job Title',
        'description_x_x': 'Job Description',
        'formatted_work_type_x': 'Work Type',
        'application_type_x': 'Application Type',
        'formatted_experience_level_x': 'Experience Level',
        'work_type_x': 'Work Category',
        'inferred_x': 'Inferred Data',
        'description_y_x': 'Company Description',
        'company_size_x': 'Company Size',
        'state_x': 'State',
        'country_x': 'Country',
        'max_salary_y': 'Max Salary',
        'pay_period_y': 'Pay Period',
        'min_salary_y': 'Min Salary',
        'remote_allowed_y': 'Remote Allowed',
        'closed_time_y': 'Closed Time',
        'compensation_type_y': 'Compensation Type',
        'type_y': 'Benefits',
        'city_y': 'City',
        'employee_count_y': 'Employee Count',
        'follower_count_y': 'Follower Count',
        'skill_name_y': 'Skills',
        'industry_name_y': 'Industries'
    }

    data.rename(columns=new_column_names, inplace=True)

    # Filter relevant data
    data = data[~data['Experience Level'].str.contains('Internship', case=False, na=False)]
    data = data[~data['Experience Level'].str.contains('Not Specified', case=False, na=False)]
    data = data[data['Pay Period'] == 'YEARLY']
    data = data[data['Country'] == 'US']

    Q1 = data['Max Salary'].quantile(0.25)
    Q3 = data['Max Salary'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data = data[(data['Max Salary'] >= 10000) & (data['Max Salary'] <= upper_bound)]

    # Convert Skills and Industries back to lists
    data['Skills'] = data['Skills'].apply(lambda x: eval(x) if isinstance(x, str) else [])
    data['Industries'] = data['Industries'].apply(lambda x: eval(x) if isinstance(x, str) else [])

    return data


from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_data(data, target_column):
    """
    Preprocesses the dataset:
    - Creates a binary target based on the 75th percentile of the target column.
    - Applies TF-IDF to 'Job Description'.
    - Label encodes other categorical features.
    - Scales numeric features.
    """

    # Create binary classification target
    threshold_salary = data[target_column].quantile(0.75)
    data[target_column] = (data[target_column] >= threshold_salary).astype(int)

    # Extract job descriptions
    job_descriptions = data['Job Description'].fillna('')

    # Drop unnecessary columns, including target, to form feature set
    X = data.drop(columns=[target_column, 'Min Salary', 'Job Title', 'Job Description'])
    y = data[target_column]

    # Label-encode other categorical features
    label_encoders = {}
    cat_cols = X.select_dtypes(include='object').columns
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Apply TF-IDF to job descriptions
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(job_descriptions)
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    # Scale numeric columns
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Combine numeric/categorical features with TF-IDF features
    X_final = pd.concat([X.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

    return X_final, y, label_encoders

def train_random_forest(X, y):
    """
    Splits the data into train-test sets, trains a Random Forest Classifier,
    and evaluates it using accuracy, precision, recall, F1-score, and a classification report.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    y_pred = rf_classifier.predict(X_test)

    print("\nAccuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
    print("Precision: {:.2f}%".format(precision_score(y_test, y_pred) * 100))
    print("Recall: {:.2f}%".format(recall_score(y_test, y_pred) * 100))
    print("F1 Score: {:.2f}%".format(f1_score(y_test, y_pred) * 100))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    feature_importances = pd.DataFrame(
        {'Feature': X.columns, 'Importance': rf_classifier.feature_importances_}
    ).sort_values(by='Importance', ascending=False)

    print("\nTop 10 Important Features:")
    print(feature_importances.head(10))

    return rf_classifier, feature_importances


def main():
    target_column = "Max Salary"
    print("Loading dataset...")
    data = load_and_clean_data('Cleaned_mini_dataset.csv')
    X, y, label_encoders = preprocess_data(data, target_column)
    rf_model = train_random_forest(X, y)


main()
