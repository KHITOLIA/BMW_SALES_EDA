import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder
import kagglehub
import seaborn as sns

# ------------------------------------------------
# Streamlit page setup
# ------------------------------------------------
st.set_page_config(
    page_title="BMW Sales EDA",
    page_icon="🚗",
    layout="wide"
)
le = LabelEncoder()

# Download latest version
path = kagglehub.dataset_download("ahmadrazakashif/bmw-worldwide-sales-records-20102024")

print("Path to dataset files:", path)

for file in os.listdir(path):
  print(file)

df = pd.read_csv(path + "/BMW sales data (2010-2024) (1).csv")

categorical_features = []
numerical_features = []
for col in df.columns:
    if df[col].dtypes == object:
        categorical_features.append(col)
    else:
        numerical_features.append(col)


def dashboard():
    st.sidebar.title("🚗 Exploratory Data Analysis ")
    page = st.sidebar.radio("📂", ["Overview", "Numerical Features Distribution", "Outlier Detection","Categorical Features Analysis"])
    if page == "Overview":
       
       st.subheader("Dataset Overview")
       st.dataframe(data = df)
       st.subheader("Statistical Summary")
       st.write(df.describe())
       st.write("### Categorical Value Counts")
       st.write(df['Model'].value_counts())
       st.write(df['Region'].value_counts())
       st.write(df['Color'].value_counts())
       st.write(df['Fuel_Type'].value_counts())
       st.write(df['Transmission'].value_counts())
       st.write("### Missing Values in Each Column")
       st.write(df.isnull().sum())

       st.write("### Categorical Features and their Unique Values")
       for col in categorical_features:
        st.write(f"{col}")
        st.write(sorted(df[col].unique()))
        st.write("")
       st.subheader("Insights")
       st.write(f'''1. Data contains 50000 rows and 11 columns
                    \n2. The dataset contains a mix of categorical and numerical features.
                    \n3. Numerical features include {numerical_features[0]}, {numerical_features[1]}, {numerical_features[2]}, {numerical_features[3]}. 
                    \n4. The dataset appears to be well-structured with no missing values in the displayed overview.
                    \n5. Categorical features include Model, Region, Color, Fuel_Type, and Transmission.
                    \n6. According to this data bwm sold their cars from 2010-2024
                    \n7. Their Engine size varies from 1.5L-5.0L
                    \n8. Price range of cars is from 20,000 USD to 150,000 USD
                    ''')
    


    elif page == "Numerical Features Distribution":
        st.subheader("Distribution of Numerical Features")
        for col in numerical_features:
            plt.figure(figsize=(8, 3))
            sns.histplot(df[col], kde=True, color="blue")
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.show()
            st.pyplot(plt)
        st.subheader("Insights")
        st.write('''1. data has been properly in the format no negative values are present here
                    \n2. All features have a uniform distribution.
                    \n3. from the above analysis we can say that no need of mileage col no relationship found.
                    \n4. hence we can drop the mileage column from the data for better analysis
                    ''')

        df.drop(columns=['Mileage_KM'], inplace=True)
        
    elif page == 'Categorical Features Analysis':
        st.subheader("Categorical Features Analysis")
        for col in categorical_features:
            plt.figure(figsize= (8, 3))
            ax = sns.countplot(x = col, data = df, palette="magma", order=df[col].value_counts().reset_index()[col])        
            plt.title(f"{col} Distribution ")
            plt.xticks(rotation = 50)
            plt.xlabel(f'{col}')
            plt.tight_layout()
            plt.show()
            st.pyplot(plt)
        st.subheader("Insights")
        st.write('''1. all categories in each categorical col are equally distributed''')

    
    elif page == "Outlier Detection":
        st.subheader("Outlier Detection using Box Plots")
        for col in numerical_features:
            plt.figure(figsize=(8, 3))
            sns.boxplot(y = col, data = df, color="orange")
            plt.title(f"{col} Distribution ")
            plt.xlabel(f'{col}')
            plt.ylabel("Values")
            plt.tight_layout()
            plt.show()
            st.pyplot(plt)
        st.subheader("Insights")
        st.write('''1. No outliers are present in the numerical features as per the box plot analysis
                    \n2. All the columns are in proper type
                    \n3. No need of outlier treatment as no outliers are present in the data
                    \n4. Data is clean and ready for further analysis or modeling
                    \n5. Engine size is mostly concentrated around 2000-3500 CC''')
    





dashboard()