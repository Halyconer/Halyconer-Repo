import pandas as pd
import numpy as np

#%%
# Load all datasets
dataset_raw = pd.read_csv('../Cleaning/Comprehensive_Data.csv') # The comprehensive dataset was made after merging several datasets
# with job data. Null values had already been dealt with.

job_postings = pd.read_csv('../Cleaning/LinkedIn Job Postings 2023-2024/postings.csv')
benefits = pd.read_csv('../Cleaning/LinkedIn Job Postings 2023-2024/jobs/benefits.csv')
job_industries = pd.read_csv('../Cleaning/LinkedIn Job Postings 2023-2024/jobs/job_industries.csv')
job_skills = pd.read_csv('../Cleaning/LinkedIn Job Postings 2023-2024/jobs/job_skills.csv')
companies = pd.read_csv('../Cleaning/LinkedIn Job Postings 2023-2024/companies/companies.csv')
employee_counts = pd.read_csv('../Cleaning/LinkedIn Job Postings 2023-2024/companies/employee_counts.csv')
company_industries = pd.read_csv('../Cleaning/LinkedIn Job Postings 2023-2024/companies/company_industries.csv')
company_specialities = pd.read_csv('../Cleaning/LinkedIn Job Postings 2023-2024/companies/company_specialities.csv')

industries_mappings = pd.read_csv('../Cleaning/LinkedIn Job Postings 2023-2024/mappings/industries.csv')
skills_mappings = pd.read_csv('../Cleaning/LinkedIn Job Postings 2023-2024/mappings/skills.csv')
industries_data = job_industries.merge(industries_mappings, on='industry_id', how='left')
skills_data = job_skills.merge(skills_mappings, on='skill_abr', how='left')

data_with_industries = pd.merge(dataset_raw, industries_data, on='job_id', how='left')
data_with_skills = pd.merge(dataset_raw, skills_data, on='job_id', how='left')

final_dataframe = pd.merge(
    data_with_skills,  # Replace with your industries dataset
    data_with_industries,      # Replace with your skills dataset
    on='job_id',
    how='outer'       # Ensures unmatched rows are included
)

mini = final_dataframe.sample(frac=0.001, random_state=42)

# Group the skills and industries into lists for each job_id
grouped_skills = mini.groupby('job_id')['skill_name'].apply(lambda x: list(x.dropna().unique())).reset_index()
grouped_industries = mini.groupby('job_id')['industry_name'].apply(lambda x: list(x.dropna().unique())).reset_index()

# Merge these lists back into the original job data, keeping unique job rows
unique_jobs = mini.drop_duplicates(subset=['job_id']).reset_index(drop=True)

# Merge skills and industries into the unique job dataset
merged_jobs = unique_jobs.merge(grouped_skills, on='job_id', how='left').merge(grouped_industries, on='job_id', how='left')

# Renaming the dataset
data = merged_jobs

#Calculate Correlations
correlations_full = data.corr(numeric_only=True)['max_salary_x'].sort_values(ascending=False)

#Identify Columns with Low Correlation (Excluding Skills and Industries)
columns_to_keep = ['industry_id', 'industry_name', 'skill_name']
low_corr_columns_excluded = [
    col for col in correlations_full.index
    if abs(correlations_full[col]) < 0.1 and col not in columns_to_keep
]

#Drop Low-Correlation Columns
cleaned_data = data.drop(columns=low_corr_columns_excluded)

#cleaned_data.to_csv('../Cleaning/Cleaned_mini_dataset_for_bootcamp.csv')