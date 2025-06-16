# %%
# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Import necessary modules for Kaggle API and file operations
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import io
import os # Imported for file operations

def pull_and_clean_spaceship_titanic_data(competition_name: str = 'spaceship-titanic') -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Pulls the Spaceship Titanic competition data from Kaggle, cleans it,
    and returns the training and testing DataFrames, along with original test PassengerIds.
    The zip file is downloaded locally temporarily and then deleted.

    Args:
        competition_name (str): The name of the Kaggle competition.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series]: A tuple containing (train_df_cleaned, test_df_cleaned, original_test_passenger_ids)
                                            cleaned DataFrames and the original PassengerIds from the test set.
                                            Returns (empty_df, empty_df, empty_series) if an error occurs.
    """
    api = KaggleApi()
    api.authenticate() # Authenticate using the kaggle.json file

    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    original_test_passenger_ids = pd.Series()
    
    # Define the temporary download path and filename
    temp_download_path = '.' 
    zip_filename = f"{competition_name}.zip"
    full_zip_path = os.path.join(temp_download_path, zip_filename)

    try:
        print(f"Downloading {competition_name} competition files to {full_zip_path}...")
        api.competition_download_cli(competition_name, path=temp_download_path, quiet=False, force=True)

        if not os.path.exists(full_zip_path):
            print(f"Error: Zip file was not downloaded to {full_zip_path}. Check permissions or network connectivity.")
            return pd.DataFrame(), pd.DataFrame(), pd.Series()

        print(f"Zip file downloaded successfully to: {full_zip_path}")

        with zipfile.ZipFile(full_zip_path, 'r') as z:
            file_list = z.namelist()
            print(f"Files found in zip: {file_list}")

            if 'train.csv' in file_list:
                with z.open('train.csv') as f:
                    train_df = pd.read_csv(f)
                print("train.csv loaded.")
            else:
                print("Error: train.csv not found in the downloaded zip.")
                return pd.DataFrame(), pd.DataFrame(), pd.Series()

            if 'test.csv' in file_list:
                with z.open('test.csv') as f:
                    test_df = pd.read_csv(f)
                print("test.csv loaded.")
            else:
                print("Error: test.csv not found in the downloaded zip.")
                return pd.DataFrame(), pd.DataFrame(), pd.Series()

        print("Data downloaded successfully. Starting cleaning...")

        original_test_passenger_ids = test_df['PassengerId'].copy()

    
        train_df_processed = train_df.drop(columns=['PassengerId', 'Name'], errors='ignore')
        test_df_processed = test_df.drop(columns=['PassengerId', 'Name'], errors='ignore')
        print("Dropped 'PassengerId' and 'Name' columns from raw train/test data.")

        combined_df = pd.concat([train_df_processed.assign(is_train=1), test_df_processed.assign(is_train=0)], ignore_index=True)


        numerical_cols_to_impute = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Age']
        for col in numerical_cols_to_impute:
            if col in combined_df.columns:
                combined_df[col] = combined_df[col].fillna(combined_df[col].median())
        print("Filled missing numerical values with median.")

        boolean_cols_to_impute = ['CryoSleep', 'VIP']
        for col in boolean_cols_to_impute:
            if col in combined_df.columns:
                combined_df[col] = combined_df[col].fillna(False)
        print("Filled missing boolean values with False.")

        categorical_cols_to_impute = ['HomePlanet', 'Destination'] 
        for col in categorical_cols_to_impute: 
            if col in combined_df.columns:
                combined_df[col] = combined_df[col].fillna('Unknown')
        print("Filled missing categorical values with 'Unknown'.")

        if 'Cabin' in combined_df.columns:
            combined_df[['Deck', 'CabinNum', 'Side']] = combined_df['Cabin'].str.split('/', expand=True)
            combined_df['Deck'] = combined_df['Deck'].fillna('Z') 
            combined_df['CabinNum'] = pd.to_numeric(combined_df['CabinNum'], errors='coerce').fillna(-1).astype(int) 
            combined_df['Side'] = combined_df['Side'].fillna('Z') 
            combined_df.drop('Cabin', axis=1, inplace=True)
            print("Split and imputed 'Cabin' feature.")

        cols_to_encode = combined_df.select_dtypes(include='object').columns.tolist()

        for col in cols_to_encode:
            combined_df[col], _ = pd.factorize(combined_df[col])
        print(f"Label encoded categorical features: {cols_to_encode}")

        train_cleaned_df = combined_df[combined_df['is_train'] == 1].drop('is_train', axis=1).copy().reset_index(drop=True)
        test_cleaned_df = combined_df[combined_df['is_train'] == 0].drop('is_train', axis=1).copy().reset_index(drop=True)

        print("Data cleaning complete.")
        return train_cleaned_df, test_cleaned_df, original_test_passenger_ids

    except Exception as e:
        print(f"An error occurred during data pulling or cleaning: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.Series()
    finally:
        if os.path.exists(full_zip_path):
            try:
                os.remove(full_zip_path)
                print(f"Cleaned up temporary zip file: {full_zip_path}")
            except Exception as e:
                print(f"Error cleaning up temporary zip file {full_zip_path}: {e}")


print("--- Starting Data Pull and Cleaning via Kaggle API ---")
train_df_cleaned, test_df_cleaned, original_test_passenger_ids = pull_and_clean_spaceship_titanic_data()

if train_df_cleaned.empty or test_df_cleaned.empty or original_test_passenger_ids.empty:
    print("Exiting: Data could not be loaded or cleaned successfully.")
else:
    print("\n--- Data Cleaning Summary (from API function) ---")
    print("Cleaned Training Data Head (features only):")
    print(train_df_cleaned.head())
    print("\nCleaned Testing Data Head (features only):")
    print(test_df_cleaned.head())

    X = train_df_cleaned.drop(columns=['Transported']) 
    y = train_df_cleaned['Transported'].astype(int) 

    non_null_mask = y.notnull()
    X = X[non_null_mask]
    y = y[non_null_mask].astype(int) 

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"\nShape of X_train: {X_train.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    print(f"Shape of X_valid: {X_valid.shape}")
    print(f"Shape of y_valid: {y_valid.shape}")

    print("\n--- Training Random Forest Classifier Model ---")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model training complete.")


    print("\n--- Evaluating Model Performance ---")
    y_pred_valid = model.predict(X_valid)
    print("Validation Accuracy:", accuracy_score(y_valid, y_pred_valid))

    print("\n--- Generating Kaggle Submission File ---")

    missing_in_test = set(X.columns) - set(test_df_cleaned.columns)
    for col in missing_in_test:
        test_df_cleaned[col] = 0 
   
    X_test_aligned = test_df_cleaned[X.columns] 

    test_predictions = model.predict(X_test_aligned).astype(bool)

    submission = pd.DataFrame({
        'PassengerId': original_test_passenger_ids, 
        'Transported': test_predictions
    })

    print("\n--- Verifying Submission File IDs ---")
    if submission['PassengerId'].duplicated().any():
        print("WARNING: Duplicate PassengerIds found in submission file!")
        print(submission[submission['PassengerId'].duplicated(keep=False)])
    else:
        print("No duplicate PassengerIds found. IDs are unique.")
    print(f"Total unique PassengerIds: {submission['PassengerId'].nunique()}")
    print(f"Total rows in submission: {len(submission)}")


    submission_filename = "submission.csv"
    submission.to_csv(submission_filename, index=False)
    print(f"Submission file '{submission_filename}' created successfully!")
    print("Remember to NOT commit submission.csv to your Git repository.")

##Graphs 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

print("\n--- Generating Model Performance Visualizations ---")

importances = model.feature_importances_
feature_names = X.columns 
forest_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12, 7))
forest_importances.plot.bar(ax=ax)
ax.set_title("Feature Importances from Random Forest Model")
ax.set_ylabel("Mean Decrease in Impurity (MDI)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout() 
plt.show()

y_test_predictions = model.predict(X_test) 

cm = confusion_matrix(y, model.predict(X), labels=[0, 1]) 

cm_valid = confusion_matrix(y_valid, y_pred_valid, labels=[0, 1]) 
disp = ConfusionMatrixDisplay(confusion_matrix=cm_valid, display_labels=['Not Transported (0)', 'Transported (1)'])

fig_cm, ax_cm = plt.subplots(figsize=(8, 8)) 
disp.plot(cmap='Blues', ax=ax_cm)
ax_cm.set_title("Confusion Matrix (Validation Set)")
plt.show()


